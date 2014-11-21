/* tsne.cc
   Jeremy Barnes, 15 January 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.

   Implementation of the t-SNE algorithm.
*/

#include "tsne.h"
#include "jml/stats/distribution.h"
#include "jml/stats/distribution_ops.h"
#include "jml/stats/distribution_simd.h"
#include "jml/utils/vector_utils.h"
#include "jml/utils/pair_utils.h"
#include "jml/utils/lightweight_hash.h"
#include "jml/algebra/matrix_ops.h"
#include "jml/arch/simd_vector.h"
#include <boost/tuple/tuple.hpp>
#include "jml/algebra/lapack.h"
#include <cmath>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include "jml/utils/worker_task.h"
#include <boost/timer.hpp>
#include "jml/arch/timers.h"
#include "jml/arch/sse2.h"
#include "jml/arch/sse2_log.h"
#include "jml/arch/cache.h"
#include "jml/utils/guard.h"
#include <boost/bind.hpp>
#include "jml/utils/environment.h"
#include "quadtree.h"
#include "vantage_point_tree.h"

using namespace std;

namespace ML {

template<typename Float>
struct V2D_Job {
    const boost::multi_array<Float, 2> & X;
    boost::multi_array<Float, 2> & D;
    const Float * sum_X;
    int i0, i1;
    
    V2D_Job(const boost::multi_array<Float, 2> & X,
            boost::multi_array<Float, 2> & D,
            const Float * sum_X,
            int i0, int i1)
        : X(X), D(D), sum_X(sum_X), i0(i0), i1(i1)
    {
    }

    void operator () ()
    {
        int d = X.shape()[1];
        
        if (d == 2) {
            unsigned i = i0;
            for (;  i + 4 <= i1;  i += 4) {
                D[i + 0][i + 0] = 0.0f;
                D[i + 1][i + 1] = 0.0f;
                D[i + 2][i + 2] = 0.0f;
                D[i + 3][i + 3] = 0.0f;
                
                for (unsigned j = 0;  j < i;  ++j) {
                    for (unsigned ii = 0;  ii < 4;  ++ii) {
                        Float XXT
                            = (X[i + ii][0] * X[j][0])
                            + (X[i + ii][1] * X[j][1]);
                        Float val = sum_X[i + ii] + sum_X[j] - 2.0f * XXT;
                        D[i + ii][j] = val;
                    }
                }
                
                // finish off the diagonal
                for (unsigned ii = 0;  ii < 4;  ++ii) {
                    for (unsigned j = i;  j < i + ii;  ++j) {
                        Float XXT
                            = (X[i + ii][0] * X[j][0])
                            + (X[i + ii][1] * X[j][1]);
                        Float val = sum_X[i + ii] + sum_X[j] - 2.0f * XXT;
                        D[i + ii][j] = val;
                    }
                }
            }
            for (;  i < i1;  ++i) {
                D[i][i] = 0.0f;
                
                for (unsigned j = 0;  j < i;  ++j) {
                    Float XXT = (X[i][0] * X[j][0]) + (X[i][1]) * (X[j][1]);
                    Float val = sum_X[i] + sum_X[j] - 2.0f * XXT;
                    D[i][j] = val;
                }
            }
        }
        else if (d < 8) {
            for (unsigned i = i0;  i < i1;  ++i) {
                D[i][i] = 0.0f;
                for (unsigned j = 0;  j < i;  ++j) {
                    float XXT = 0.0;
                    for (unsigned k = 0;  k < d;  ++k)
                        XXT += X[i][k] * X[j][k];
                    
                    Float val = sum_X[i] + sum_X[j] - 2.0f * XXT;
                    D[i][j] = val;
                }
            }
        }
        else {
            for (unsigned i = i0;  i < i1;  ++i) {
                D[i][i] = 0.0f;
                for (unsigned j = 0;  j < i;  ++j) {
                    // accum in double precision for accuracy
                    Float XXT = SIMD::vec_dotprod_dp(&X[i][0], &X[j][0], d);
                    Float val = sum_X[i] + sum_X[j] - 2.0f * XXT;
                    D[i][j] = val;
                }
            }
        }
    }
};

template<typename Float>
void
vectors_to_distances(const boost::multi_array<Float, 2> & X,
                     boost::multi_array<Float, 2> & D,
                     bool fill_upper)
{
    // again, ||y_i - y_j||^2 
    //     = sum_d ( y_id - y_jd )^2
    //     = sum_d ( y_id^2 + y_jd^2 - 2 y_id y_jd)
    //     = sum_d ( y_id^2) + sum_d(y_jd^2) - 2 sum_d(y_id y_jd)
    //     = ||y_i||^2 + ||y_j||^2 - 2 sum_d(y_id y_jd)
    
    int n = X.shape()[0];

    if (D.shape()[0] != n || D.shape()[1] != n)
        throw Exception("D matrix should be square with (n x n) shape");
    
    int d = X.shape()[1];

    distribution<Float> sum_X(n);

    if (d < 16) {
        for (unsigned i = 0;  i < n;  ++i) {
            double total = 0.0;  // accum in double precision for accuracy
            for (unsigned j = 0;  j < d;  ++j)
                total += X[i][j] * X[i][j];
            sum_X[i] = total;
        }
    }
    else {
        for (unsigned i = 0;  i < n;  ++i)
            sum_X[i] = SIMD::vec_dotprod_dp(&X[i][0], &X[i][0], d);
    }
    
    Worker_Task & worker = Worker_Task::instance(num_threads() - 1);

    int group;
    {
        int parent = -1;  // no parent group
        group = worker.get_group(NO_JOB, "", parent);
        Call_Guard guard(boost::bind(&Worker_Task::unlock_group,
                                     boost::ref(worker),
                                     group));
        
        int chunk_size = 256;
        
        for (int i = n;  i > 0;  i -= chunk_size) {
            int i0 = max(0, i - chunk_size);
            int i1 = i;
            
            worker.add(V2D_Job<Float>(X, D, &sum_X[0], i0, i1),
                       "", group);
        }
    }
    
    worker.run_until_finished(group);

    if (fill_upper)
        copy_lower_to_upper(D);
}

void
vectors_to_distances(const boost::multi_array<float, 2> & X,
                     boost::multi_array<float, 2> & D,
                     bool fill_upper)
{
    return vectors_to_distances<float>(X, D, fill_upper);
}

void
vectors_to_distances(const boost::multi_array<double, 2> & X,
                     boost::multi_array<double, 2> & D,
                     bool fill_upper)
{
    return vectors_to_distances<double>(X, D, fill_upper);
}

template<typename Float>
double
perplexity(const distribution<Float> & p)
{
    double total = 0.0;
    for (unsigned i = 0;  i < p.size();  ++i)
        if (p[i] != 0.0) total -= p[i] * log(p[i]);
    return exp(total);
}

/** Compute the perplexity and the P for a given value of beta. */
template<typename Float>
std::pair<double, distribution<Float> >
perplexity_and_prob(const distribution<Float> & D, double beta = 1.0,
                    int i = -1)
{
    distribution<Float> P(D.size());
    SIMD::vec_exp(&D[0], -beta, &P[0], D.size());
    if (i != -1) P[i] = 0;
    double tot = P.total();

    if (!isfinite(tot) || tot == 0) {
        // Perplexity is impossible, since no weights
        std::fill(P.begin(), P.end(), 1.0);
        if (i != -1)
            P.at(i) = 0;
        P.normalize();
        return make_pair(INFINITY, P);
#if 1
        cerr << "beta = " << beta << endl;
        cerr << "D = " << D << endl;
        cerr << "tot = " << tot << endl;
        cerr << "i = " << i << endl;
        cerr << "P = " << P << endl;
#endif
        throw Exception("non-finite total for perplexity");
    }

    double H = log(tot) + beta * D.dotprod(P) / tot;
    P *= 1.0 / tot;

    if (!isfinite(P.total())) {
#if 1
        cerr << "beta = " << beta << endl;
        cerr << "D = " << D << endl;
        cerr << "tot = " << tot << endl;
        cerr << "i = " << i << endl;
#endif
        throw Exception("non-finite total for perplexity");
    }


    return make_pair(H, P);
}

std::pair<double, distribution<float> >
perplexity_and_prob(const distribution<float> & D, double beta,
                    int i)
{
    return perplexity_and_prob<float>(D, beta, i);
}

std::pair<double, distribution<double> >
perplexity_and_prob(const distribution<double> & D, double beta,
                    int i)
{
    return perplexity_and_prob<double>(D, beta, i);
}


/** Calculate the beta for a single point.
    
    \param Di     The i-th row of the D matrix, for which we want to calculate
                  the probabilities.
    \param i      Which row number it is.

    \returns      The i-th row of the P matrix, which has the distances in D
                  converted to probabilities with the given perplexity.
 */
std::pair<distribution<float>, double>
binary_search_perplexity(const distribution<float> & Di,
                         double required_perplexity,
                         int i,
                         double tolerance)
{
    double betamin = -INFINITY, betamax = INFINITY;
    double beta = 1.0;

    distribution<float> P;
    double log_perplexity;
    double log_required_perplexity = log(required_perplexity);

    boost::tie(log_perplexity, P) = perplexity_and_prob(Di, beta, i);

    if (log_perplexity == INFINITY) {
        // Ill conditioned, there is nothing to do
        return make_pair(P, INFINITY);
    }

    bool verbose = false;

    if (verbose)
        cerr << "iter currperp targperp     diff toleranc   betamin     beta  betamax" << endl;
    
    for (unsigned iter = 0;  iter != 50;  ++iter) {
        if (verbose) 
            cerr << format("%4d %8.4f %8.4f %8.4f %8.4f  %8.4f %8.4f %8.4f\n",
                           iter,
                           log_perplexity, log_required_perplexity,
                           fabs(log_perplexity - log_required_perplexity),
                           tolerance,
                           betamin, beta, betamax);
        
        if (fabs(log_perplexity - log_required_perplexity) < tolerance)
            break;

        if (log_perplexity > log_required_perplexity) {
            betamin = beta;
            if (!isfinite(betamax))
                beta *= 2;
            else beta = (beta + betamax) * 0.5;
        }
        else {
            betamax = beta;
            if (!isfinite(betamin))
                beta /= 2;
            else beta = (beta + betamin) * 0.5;
        }
        
        boost::tie(log_perplexity, P) = perplexity_and_prob(Di, beta, i);
    }

    return make_pair(P, beta);
}

struct Distance_To_Probabilities_Job {

    boost::multi_array<float, 2> & D;
    double tolerance;
    double perplexity;
    boost::multi_array<float, 2> & P;
    distribution<float> & beta;
    int i0;
    int i1;

    Distance_To_Probabilities_Job(boost::multi_array<float, 2> & D,
                                  double tolerance,
                                  double perplexity,
                                  boost::multi_array<float, 2> & P,
                                  distribution<float> & beta,
                                  int i0,
                                  int i1)
        : D(D), tolerance(tolerance), perplexity(perplexity),
          P(P), beta(beta), i0(i0), i1(i1)
    {
    }

    void operator () ()
    {
        int n = D.shape()[0];

        for (unsigned i = i0;  i < i1;  ++i) {
            //cerr << "i = " << i << endl;
            //if (i % 250 == 0)
            //    cerr << "P-values for point " << i << " of " << n << endl;
            
            distribution<float> D_row(&D[i][0], &D[i][0] + n);
            distribution<float> P_row;

            try {
                boost::tie(P_row, beta[i])
                    = binary_search_perplexity(D_row, perplexity, i, tolerance);
            } catch (const std::exception & exc) {
                P_row = D_row;
                P_row[i] = 1000000;
                P_row = (P_row == P_row.min());
                std::fill(P_row.begin(), P_row.end(), 1.0);
                P_row[i] = 0.0;
                P_row.normalize();
            }
            
            if (P_row.size() != n)
                throw Exception("P_row has the wrong size");
            if (P_row[i] != 0.0) {
                cerr << "i = " << i << endl;
                //cerr << "D_row = " << D_row << endl;
                //cerr << "P_row = " << P_row << endl;
                cerr << "P_row.total() = " << P_row.total() << endl;
                cerr << "P_row[i] = " << P_row[i] << endl;
                throw Exception("P_row diagonal entry was not zero");
            }

            std::copy(P_row.begin(), P_row.end(), &P[i][0]);
        }
    }
};


/* Given a matrix of distances, convert to probabilities */
boost::multi_array<float, 2>
distances_to_probabilities(boost::multi_array<float, 2> & D,
                           double tolerance,
                           double perplexity)
{
    int n = D.shape()[0];
    if (D.shape()[1] != n)
        throw Exception("D is not square");

    boost::multi_array<float, 2> P(boost::extents[n][n]);
    distribution<float> beta(n, 1.0);

    Worker_Task & worker = Worker_Task::instance(num_threads() - 1);

    int group;
    {
        int parent = -1;  // no parent group
        group = worker.get_group(NO_JOB, "", parent);
        Call_Guard guard(boost::bind(&Worker_Task::unlock_group,
                                     boost::ref(worker),
                                     group));
        
        int chunk_size = 256;
        
        for (int i = 0;  i < n;  i += chunk_size) {
            int i0 = i;
            int i1 = min(n, i + chunk_size);
            
            worker.add(Distance_To_Probabilities_Job
                       (D, tolerance, perplexity, P, beta, i0, i1),
                       "", group);
        }
    }

    worker.run_until_finished(group);

    cerr << "mean sigma is " << sqrt(1.0 / beta).mean() << endl;

    return P;
}

boost::multi_array<float, 2>
pca(boost::multi_array<float, 2> & coords, int num_dims)
{
    // TODO: normalize the input coordinates (especially if it seems to be
    // ill conditioned)

    int nx = coords.shape()[0];
    int nd = coords.shape()[1];

    int nvalues = std::min(nd, nx);

    int ndr = std::min(nvalues, num_dims);

    if (ndr < num_dims)
        throw Exception("svd_reduction: num_dims not low enough");
        
    distribution<float> svalues(nvalues);
    boost::multi_array<float, 2> lvectorsT(boost::extents[nvalues][nd]);
    boost::multi_array<float, 2> rvectors(boost::extents[nx][nvalues]);

    int res = LAPack::gesdd("S", nd, nx,
                            coords.data(), nd,
                            &svalues[0],
                            &lvectorsT[0][0], nd,
                            &rvectors[0][0], nvalues);
    
    // If some vectors are singular, ignore them
    // TODO: do...
        
    if (res != 0)
        throw Exception("gesdd returned non-zero");
        
    boost::multi_array<float, 2> result(boost::extents[nx][ndr]);
    for (unsigned i = 0;  i < nx;  ++i)
        std::copy(&rvectors[i][0], &rvectors[i][0] + ndr, &result[i][0]);

    return result;
}

double calc_D_row(float * Di, int n)
{
    unsigned i = 0;

    double total = 0.0;

    if (false) ;
    else if (n >= 8) {
        using namespace SIMD;

        v2df rr = vec_splat(0.0);
        
        v4sf one = vec_splat(1.0f);

        __builtin_prefetch(Di + i + 0, 1, 3);
        __builtin_prefetch(Di + i + 16, 1, 3);
        __builtin_prefetch(Di + i + 32, 1, 3);

        for (; i + 16 <= n;  i += 16) {
            __builtin_prefetch(Di + i + 48, 1, 3);

            v4sf xxxx0 = __builtin_ia32_loadups(Di + i + 0);
            v4sf xxxx1 = __builtin_ia32_loadups(Di + i + 4);
            xxxx0      = xxxx0 + one;
            xxxx1      = xxxx1 + one;
            xxxx0      = one / xxxx0;
            v4sf xxxx2 = __builtin_ia32_loadups(Di + i + 8);
            xxxx1      = one / xxxx1;
            __builtin_ia32_storeups(Di + i + 0, xxxx0);
            xxxx2      = xxxx2 + one;
            v2df xx0a, xx0b;  vec_f2d(xxxx0, xx0a, xx0b);
            __builtin_ia32_storeups(Di + i + 4, xxxx1);
            xx0a       = xx0a + xx0b;
            rr         = rr + xx0a;
            v4sf xxxx3 = __builtin_ia32_loadups(Di + i + 12);
            v2df xx1a, xx1b;  vec_f2d(xxxx1, xx1a, xx1b);
            xxxx2      = one / xxxx2;
            xx1a       = xx1a + xx1b;
            __builtin_ia32_storeups(Di + i + 8, xxxx2);
            rr         = rr + xx1a;
            v2df xx2a, xx2b;  vec_f2d(xxxx2, xx2a, xx2b);
            xxxx3      = xxxx3 + one;
            xx2a       = xx2a + xx2b;
            xxxx3      = one / xxxx3;
            rr         = rr + xx2a;
            v2df xx3a, xx3b;  vec_f2d(xxxx3, xx3a, xx3b);
            __builtin_ia32_storeups(Di + i + 12, xxxx3);
            xx3a       = xx3a + xx3b;
            rr         = rr + xx3a;
        }

        for (; i + 4 <= n;  i += 4) {
            v4sf xxxx0 = __builtin_ia32_loadups(Di + i + 0);
            xxxx0      = xxxx0 + one;
            xxxx0      = one / xxxx0;
            __builtin_ia32_storeups(Di + i + 0, xxxx0);

            v2df xx0a, xx0b;
            vec_f2d(xxxx0, xx0a, xx0b);

            rr      = rr + xx0a;
            rr      = rr + xx0b;
        }

        double results[2];
        *(v2df *)results = rr;

        total = (results[0] + results[1]);
    }
    
    for (;  i < n;  ++i) {
        Di[i] = 1.0f / (1.0f + Di[i]);
        total += Di[i];
    }

    return total;
}

namespace {

Env_Option<bool> PROFILE_TSNE("PROFILE_TSNE", false);

double t_v2d = 0.0, t_D = 0.0, t_dY = 0.0, t_update = 0.0;
double t_recenter = 0.0, t_cost = 0.0, t_PmQxD = 0.0, t_clu = 0.0;
double t_stiffness = 0.0;
struct AtEnd {
    ~AtEnd()
    {
        if (!PROFILE_TSNE) return;

        cerr << "tsne core profile:" << endl;
        cerr << "  v2d:        " << t_v2d << endl;
        cerr << "  stiffness:" << t_stiffness << endl;
        cerr << "    D         " << t_D << endl;
        cerr << "    (P-Q)D    " << t_PmQxD << endl;
        cerr << "    clu       " << t_clu << endl;
        cerr << "  dY:         " << t_dY << endl;
        cerr << "  update:     " << t_update << endl;
        cerr << "  recenter:   " << t_recenter << endl;
        cerr << "  cost:       " << t_cost << endl;
    }
} atend;

} // file scope

struct Calc_D_Job {

    boost::multi_array<float, 2> & D;
    int i0;
    int i1;
    double * d_totals;

    Calc_D_Job(boost::multi_array<float, 2> & D,
               int i0,
               int i1,
               double * d_totals)
        : D(D), i0(i0), i1(i1), d_totals(d_totals)
    {
    }

    void operator () ()
    {
        for (unsigned i = i0;  i < i1;  ++i) {
            d_totals[i] = 2.0 * calc_D_row(&D[i][0], i);
            D[i][i] = 0.0f;
        }
    }
};

double calc_stiffness_row(float * Di, const float * Pi, float qfactor,
                          float min_prob, int n, bool calc_costs)
{
    double cost = 0.0;

    unsigned i = 0;

    if (false) ;
    else if (true) {
        using namespace SIMD;

        v4sf mmmm = vec_splat(min_prob);
        v4sf ffff = vec_splat(qfactor);

        v2df total = vec_splat(0.0);

        for (; i + 4 <= n;  i += 4) {

            v4sf dddd0 = __builtin_ia32_loadups(Di + i + 0);
            v4sf pppp0 = __builtin_ia32_loadups(Pi + i + 0);
            v4sf qqqq0 = __builtin_ia32_maxps(mmmm, dddd0 * ffff);
            v4sf ssss0 = (pppp0 - qqqq0) * dddd0;
            __builtin_ia32_storeups(Di + i + 0, ssss0);
            if (JML_LIKELY(!calc_costs)) continue;

            v4sf pqpq0  = pppp0 / qqqq0;
            v4sf lpq0   = sse2_logf_unsafe(pqpq0);
            v4sf cccc0  = pppp0 * lpq0;
            cccc0 = cccc0 + cccc0;

            v2df cc0a, cc0b;
            vec_f2d(cccc0, cc0a, cc0b);

            total   = total + cc0a;
            total   = total + cc0b;
        }

        double results[2];
        *(v2df *)results = total;
        
        cost = results[0] + results[1];
    }

    for (;  i < n;  ++i) {
        float d = Di[i];
        float p = Pi[i];
        float q = std::max(min_prob, d * qfactor);
        Di[i] = (p - q) * d;
        if (calc_costs) cost += 2.0 * p * logf(p / q);
    }

    return cost;
}

struct Calc_Stiffness_Job {

    boost::multi_array<float, 2> & D;
    const boost::multi_array<float, 2> & P;
    float min_prob;
    float qfactor;
    double * costs;
    int i0, i1;

    Calc_Stiffness_Job(boost::multi_array<float, 2> & D,
                       const boost::multi_array<float, 2> & P,
                       float min_prob,
                       float qfactor,
                       double * costs,
                       int i0, int i1)
        : D(D), P(P), min_prob(min_prob),
          qfactor(qfactor), costs(costs), i0(i0), i1(i1)
    {
    }

    void operator () ()
    {
        for (unsigned i = i0;  i < i1;  ++i) {
            double cost 
                = calc_stiffness_row(&D[i][0], &P[i][0],
                                     qfactor, min_prob, i,
                                     costs);
            if (costs) costs[i] = cost;
        }
    }
};

double tsne_calc_stiffness(boost::multi_array<float, 2> & D,
                           const boost::multi_array<float, 2> & P,
                           float min_prob,
                           bool calc_cost)
{
    boost::timer t;

    int n = D.shape()[0];
    if (D.shape()[1] != n)
        throw Exception("D has wrong shape");

    if (P.shape()[0] != n || P.shape()[1] != n)
        throw Exception("P has wrong shape");

    double d_totals[n];

    Worker_Task & worker = Worker_Task::instance(num_threads() - 1);

    int group;
    {
        int parent = -1;  // no parent group
        group = worker.get_group(NO_JOB, "", parent);
        Call_Guard guard(boost::bind(&Worker_Task::unlock_group,
                                     boost::ref(worker),
                                     group));
        
        int chunk_size = 256;
        
        for (int i = n;  i > 0;  i -= chunk_size) {
            int i0 = max(0, i - chunk_size);
            int i1 = i;
            
            worker.add(Calc_D_Job(D, i0, i1, d_totals),
                       "", group);
        }
    }
    
    worker.run_until_finished(group);

    double d_total_offdiag = SIMD::vec_sum(d_totals, n);

    t_D += t.elapsed();  t.restart();
    
    // Cost accumulated for each row
    double row_costs[n];

    // Q matrix: q_{i,j} = d_{ij} / sum_{k != l} d_{kl}
    float qfactor = 1.0 / d_total_offdiag;

    {
        int parent = -1;  // no parent group
        group = worker.get_group(NO_JOB, "", parent);
        Call_Guard guard(boost::bind(&Worker_Task::unlock_group,
                                     boost::ref(worker),
                                     group));
        
        int chunk_size = 64;
        
        for (int i = n;  i > 0;  i -= chunk_size) {
            int i0 = max(0, i - chunk_size);
            int i1 = i;
            
            worker.add(Calc_Stiffness_Job
                       (D, P, min_prob, qfactor,
                        (calc_cost ? row_costs : (double *)0), i0, i1),
                       "", group);
        }
    }

    worker.run_until_finished(group);

    double cost = 0.0;
    if (calc_cost) cost = SIMD::vec_sum(row_costs, n);

    t_PmQxD += t.elapsed();  t.restart();
    
    copy_lower_to_upper(D);
    
    t_clu += t.elapsed();  t.restart();

    return cost;
}

inline void
calc_dY_rows_2d(boost::multi_array<float, 2> & dY,
                const boost::multi_array<float, 2> & PmQxD,
                const boost::multi_array<float, 2> & Y,
                int i, int n)
{
#if 1
    using namespace SIMD;

    v4sf totals01 = vec_splat(0.0f), totals23 = totals01;
    v4sf four = vec_splat(4.0f);

    for (unsigned j = 0;  j < n;  ++j) {
        //v4sf ffff = { PmQxD[i + 0][j], PmQxD[i + 1][j],
        //              PmQxD[i + 2][j], PmQxD[i + 3][j] };
        // TODO: expand inplace

        v4sf ffff01 = { PmQxD[i + 0][j], PmQxD[i + 0][j],
                        PmQxD[i + 1][j], PmQxD[i + 1][j] };
        v4sf ffff23 = { PmQxD[i + 2][j], PmQxD[i + 2][j],
                        PmQxD[i + 3][j], PmQxD[i + 3][j] };

        // TODO: load once and shuffle into position
        v4sf yjyj   = { Y[j][0], Y[j][1], Y[j][0], Y[j][1] };

        ffff01 = ffff01 * four;
        ffff23 = ffff23 * four;
        
        v4sf yi01   = __builtin_ia32_loadups(&Y[i][0]);
        v4sf yi23   = __builtin_ia32_loadups(&Y[i + 2][0]);

        v4sf xxxx01 = ffff01 * (yi01 - yjyj);
        v4sf xxxx23 = ffff23 * (yi23 - yjyj);
        
        totals01 += xxxx01;
        totals23 += xxxx23;
    }

    __builtin_ia32_storeups(&dY[i][0], totals01);
    __builtin_ia32_storeups(&dY[i + 2][0], totals23);

#else
    enum { b = 4 };

    float totals[b][2];
    for (unsigned ii = 0;  ii < b;  ++ii)
        totals[ii][0] = totals[ii][1] = 0.0f;
            
    for (unsigned j = 0;  j < n;  ++j) {
        float Yj0 = Y[j][0];
        float Yj1 = Y[j][1];
        
        for (unsigned ii = 0;  ii < b;  ++ii) {
            float factor = 4.0f * PmQxD[i + ii][j];
            totals[ii][0] += factor * (Y[i + ii][0] - Yj0);
            totals[ii][1] += factor * (Y[i + ii][1] - Yj1);
        }
    }
    
    for (unsigned ii = 0;  ii < b;  ++ii) {
        dY[i + ii][0] = totals[ii][0];
        dY[i + ii][1] = totals[ii][1];
    }
#endif
}

inline void
calc_dY_row_2d(float * dYi, const float * PmQxDi,
               const boost::multi_array<float, 2> & Y,
               int i,
               int n)
{
    float total0 = 0.0f, total1 = 0.0f;
    for (unsigned j = 0;  j < n;  ++j) {
        float factor = 4.0f * PmQxDi[j];
        total0 += factor * (Y[i][0] - Y[j][0]);
        total1 += factor * (Y[i][1] - Y[j][1]);
    }
    
    dYi[0] = total0;
    dYi[1] = total1;
}


struct Calc_Gradient_Job {
    boost::multi_array<float, 2> & dY;
    const boost::multi_array<float, 2> & Y;
    const boost::multi_array<float, 2> & PmQxD;
    int i0, i1;

    Calc_Gradient_Job(boost::multi_array<float, 2> & dY,
                      const boost::multi_array<float, 2> & Y,
                      const boost::multi_array<float, 2> & PmQxD,
                      int i0,
                      int i1)
        : dY(dY),
          Y(Y),
          PmQxD(PmQxD),
          i0(i0),
          i1(i1)
    {
    }
    
    void operator () ()
    {
        int n = Y.shape()[0];
        int d = Y.shape()[1];

        if (d == 2) {
            unsigned i = i0;
            
            for (;  i + 4 <= i1;  i += 4)
                calc_dY_rows_2d(dY, PmQxD, Y, i, n);
            
            for (; i < i1;  ++i)
                calc_dY_row_2d(&dY[i][0], &PmQxD[i][0], Y, i, n);
        }
        else {
            for (unsigned i = i0;  i < i1;  ++i) {
                for (unsigned k = 0;  k < d;  ++k) {
                    float Yik = Y[i][k];
                    float total = 0.0;
                    for (unsigned j = 0;  j < n;  ++j) {
                        float factor = 4.0f * PmQxD[i][j];
                        float Yjk = Y[j][k];
                        total += factor * (Yik - Yjk);
                    }
                    dY[i][k] = total;
                }
            }
        }
    }
};


void tsne_calc_gradient(boost::multi_array<float, 2> & dY,
                        const boost::multi_array<float, 2> & Y,
                        const boost::multi_array<float, 2> & PmQxD)
{
    // Gradient
    // Implements formula 5 in (Van der Maaten and Hinton, 2008)
    // dC/dy_i = 4 * sum_j ( (p_ij - q_ij)(y_i - y_j)d_ij )

    
    int n = Y.shape()[0];
    int d = Y.shape()[1];
    
    if (dY.shape()[0] != n || dY.shape()[1] != d)
        throw Exception("dY matrix has wrong shape");

    if (PmQxD.shape()[0] != n || PmQxD.shape()[1] != n)
        throw Exception("PmQxD matrix has wrong shape");

    Worker_Task & worker = Worker_Task::instance(num_threads() - 1);

    int group;
    {
        int parent = -1;  // no parent group
        group = worker.get_group(NO_JOB, "", parent);
        Call_Guard guard(boost::bind(&Worker_Task::unlock_group,
                                     boost::ref(worker),
                                     group));
        
        int chunk_size = 64;
        
        for (unsigned i = 0;  i < n;  i += chunk_size) {
            int i0 = i;
            int i1 = min(i0 + chunk_size, n);
            
            worker.add(Calc_Gradient_Job(dY, Y, PmQxD, i0, i1),
                       "", group);
        }
    }
    
    worker.run_until_finished(group);
}

void tsne_update(boost::multi_array<float, 2> & Y,
                 boost::multi_array<float, 2> & dY,
                 boost::multi_array<float, 2> & iY,
                 boost::multi_array<float, 2> & gains,
                 bool first_iter,
                 float momentum,
                 float eta,
                 float min_gain)
{
    int n = Y.shape()[0];
    int d = Y.shape()[1];

    // Implement scheme in Jacobs, 1988.  If we go in the same direction as
    // last time, we increase the learning speed of the parameter a bit.
    // If on the other hand the direction changes, we reduce exponentially
    // the rate.
    
    for (unsigned i = 0;  !first_iter && i < n;  ++i) {
        // We use != here as we gradients in dY are the negatives of what
        // we want.
        for (unsigned j = 0;  j < d;  ++j) {
            if (dY[i][j] * iY[i][j] < 0.0f)
                gains[i][j] = gains[i][j] + 0.2f;
            else gains[i][j] = gains[i][j] * 0.8f;
            gains[i][j] = std::max(min_gain, gains[i][j]);
        }
    }

    for (unsigned i = 0;  i < n;  ++i)
        for (unsigned j = 0;  j < d;  ++j)
            iY[i][j] = momentum * iY[i][j] - (eta * gains[i][j] * dY[i][j]);
    Y = Y + iY;
}
    
template<typename Float>
void recenter_about_origin(boost::multi_array<Float, 2> & Y)
{
    int n = Y.shape()[0];
    int d = Y.shape()[1];

    // Recenter Y values about the origin
    double Y_means[d];
    std::fill(Y_means, Y_means + d, 0.0);
    for (unsigned i = 0;  i < n;  ++i)
        for (unsigned j = 0;  j < d;  ++j)
            Y_means[j] += Y[i][j];
    
    Float n_recip = 1.0f / n;
    
    for (unsigned i = 0;  i < n;  ++i)
        for (unsigned j = 0;  j < d;  ++j)
            Y[i][j] -= Y_means[j] * n_recip;
}

boost::multi_array<float, 2>
tsne_init(int nx, int nd, int randomSeed)
{
    boost::mt19937 rng;
    if (randomSeed)
        rng.seed(randomSeed);
    boost::normal_distribution<float> norm;

    boost::variate_generator<boost::mt19937,
                             boost::normal_distribution<float> >
        randn(rng, norm);

    boost::multi_array<float, 2> Y(boost::extents[nx][nd]);
    for (unsigned i = 0;  i < nx;  ++i)
        for (unsigned j = 0;  j < nd;  ++j)
            Y[i][j] = 0.0001 * randn();

    return Y;
}

boost::multi_array<float, 2>
tsne(const boost::multi_array<float, 2> & probs,
     int num_dims,
     const TSNE_Params & params,
     const TSNE_Callback & callback)
{
    int n = probs.shape()[0];
    if (n != probs.shape()[1])
        throw Exception("probabilities were the wrong shape");

    int d = num_dims;

    // Coordinates
    boost::multi_array<float, 2> Y = tsne_init(n, d, params.randomSeed);

    // Symmetrize and probabilize P
    boost::multi_array<float, 2> P = probs + transpose(probs);

    // TODO: symmetric so only need to total the upper diagonal
    double sumP = 0.0;
    for (unsigned i = 0;  i < n;  ++i)
        sumP += 2.0 * SIMD::vec_sum_dp(&P[i][0], i);
    
    // Factor that P should be multiplied by in all calculations
    // We boost it by 4 in early iterations to force the clusters to be
    // spread apart
    float pfactor = 4.0 / sumP;

    // TODO: do we need this?   P = Math.maximum(P, 1e-12);
    for (unsigned i = 0;  i < n;  ++i)
        for (unsigned j = 0;  j < n;  ++j)
            P[i][j] = std::max((i != j) * pfactor * P[i][j], 1e-12f);

    double sump0 = 0.0;
    for (unsigned i = 0;  i < n;  ++i) {
        sump0 += P[0][i];
    }

    cerr << "sump0 = " << sump0 << endl;

    double sump1 = 0.0;
    for (unsigned i = 0;  i < n;  ++i) {
        sump1 += P[1][i];
    }

    cerr << "sump1 = " << sump1 << endl;

    Timer timer;

    // Pseudo-distance array for reduced space.  Q = D * qfactor
    boost::multi_array<float, 2> D(boost::extents[n][n]);

    // Y delta
    boost::multi_array<float, 2> dY(boost::extents[n][d]);

    // Last change in Y; so that we can see if we're going in the same dir
    boost::multi_array<float, 2> iY(boost::extents[n][d]);

    // Per-variable factors to multiply the gradient by to improve convergence
    boost::multi_array<float, 2> gains(boost::extents[n][d]);
    std::fill(gains.data(), gains.data() + gains.num_elements(), 1.0f);


    double cost = INFINITY;
    double last_cost = INFINITY;
    
    if (callback
        && !callback(-1, cost, "init")) return Y;

    for (int iter = 0;  iter < params.max_iter;  ++iter) {

        boost::timer t;

        /*********************************************************************/
        // Pairwise affinities Qij
        // Implements formula 4 in (Van der Maaten and Hinton, 2008)
        // q_{ij} = d_{ij} / sum_{k,l, k != l} d_{kl}
        // where d_{ij} = 1 / (1 + ||y_i - y_j||^2)

        // TODO: these will all be symmetric; we could save lots of work by
        // using upper/lower diagonal matrices.

        vectors_to_distances(Y, D, false /* fill_upper */);

        t_v2d += t.elapsed();  t.restart();
        
        if (callback
            && !callback(iter, cost, "v2d")) return Y;

        // Do we calculate the cost?
        bool calc_cost = iter < 10 || (iter + 1) % 10 == 0 || iter == params.max_iter - 1;
        
        double cost2 = tsne_calc_stiffness(D, P, params.min_prob, calc_cost);
        if (calc_cost) {
            last_cost = cost;
            cost = cost2;

            if (isfinite(cost) && cost == last_cost) {
                // converged
                break;
            }
                
        }

        if (callback
            && !callback(iter, cost, "stiffness")) return Y;

        t_stiffness += t.elapsed();  t.restart();

        // D is now the stiffness
        const boost::multi_array<float, 2> & stiffness = D;

        
        /*********************************************************************/
        // Gradient
        // Implements formula 5 in (Van der Maaten and Hinton, 2008)
        // dC/dy_i = 4 * sum_j ( (p_ij - q_ij)(y_i - y_j)d_ij )

        tsne_calc_gradient(dY, Y, stiffness);

        t_dY += t.elapsed();  t.restart();

        if (callback
            && !callback(iter, cost, "gradient")) return Y;

#if 1
        cerr << "C = " << cost << endl;
        for (unsigned x = 0;  x < 5;  ++x) {

            cerr << "P[" << x << "][0..5] = "
                 << P[x][0]
                 << " " << P[x][1]
                 << " " << P[x][2]
                 << " " << P[x][3]
                 << " " << P[x][4]
                 << endl;

            for (unsigned i = 0;  i < d;  ++i) {
                    cerr << "dY[" << x << "][" << i << "]: real " << dY[x][i]
                         << endl;
                    cerr << "Y = " << Y[x][i] << endl;
            }
        }

        return Y;
#endif

        /*********************************************************************/
        // Update

        float momentum = (iter < 20
                          ? params.initial_momentum
                          : params.final_momentum);

        tsne_update(Y, dY, iY, gains, iter == 0, momentum, params.eta,
                    params.min_gain);

        if (callback
            && !callback(iter, cost, "update")) return Y;

        t_update += t.elapsed();  t.restart();


        /*********************************************************************/
        // Recenter about the origin

        recenter_about_origin(Y);

        if (callback
            && !callback(iter, cost, "recenter")) return Y;

        t_recenter += t.elapsed();  t.restart();


        /*********************************************************************/
        // Calculate cost

        if ((iter + 1) % 100 == 0 || iter == params.max_iter - 1) {
            cerr << format("iteration %4d cost %6.3f  ",
                           iter + 1, cost)
                 << timer.elapsed() << endl;
            timer.restart();
        }
        
        t_cost += t.elapsed();  t.restart();

        // Stop lying about P values if we're finished
        if (iter == 100) {
            for (unsigned i = 0;  i < n;  ++i)
                for (unsigned j = 0;  j < n;  ++j)
                    P[i][j] *= 0.25f;
        }
    }

    return Y;
}

double sqr(double d)
{
    return d * d;
}

float pythag_dist(const float * d1, const float * d2, int nd)
{
    float diff[nd];
    SIMD::vec_add(d1, -1.0f, d2, diff, nd);
    return sqrtf(SIMD::vec_dotprod_dp(diff, diff, nd));
}

std::vector<TsneSparseProbs>
sparseProbsFromCoords(const boost::multi_array<float, 2> & coords,
                      int numNeighbours,
                      double perplexity,
                      double tolerance,
                      std::unique_ptr<VantagePointTree> * treeOut)
{
    int nx = coords.shape()[0];
    int nd = coords.shape()[1];

    // Distance between neighbours.  Must satisfy the triangle inequality,
    // so the sqrt is important.
    auto dist = [&] (int x1, int x2)
        {
            return pythag_dist(&coords[x1][0], &coords[x2][0], nd);
        };

    std::vector<int> examples;
    for (unsigned i = 0;  i < nx;  ++i)
        examples.push_back(i);

    std::unique_ptr<VantagePointTree> tree
        (VantagePointTree::create(examples, dist));

    // For each one, find the numNeighbours nearest neighbours
    std::vector<TsneSparseProbs> neighbours(nx);

    auto calcExample = [&] (int x)
        {
            neighbours[x]
                = sparseProbsFromCoords(coords, &coords[x][0], *tree, numNeighbours,
                                        perplexity, tolerance, true /* remove one */);
        };

    run_in_parallel_blocked(0, nx, calcExample);

    if (treeOut)
        treeOut->reset(tree.release());

    return neighbours;
}

TsneSparseProbs
sparseProbsFromCoords(const boost::multi_array<float, 2> & coords,
                      const float * newExampleCoords,
                      const VantagePointTree & tree,
                      int numNeighbours,
                      double perplexity,
                      double tolerance,
                      bool removeOne)
{
    int nd = coords.shape()[1];

    // Distance between neighbours.  Must satisfy the triangle inequality,
    // so the sqrt is important.
    auto dist = [&] (int x2)
        {
            return pythag_dist(&newExampleCoords[0], &coords[x2][0], nd);
        };

    TsneSparseProbs result;

    // Find the nearest neighbours
    std::vector<std::pair<float, int> > exNeighbours
        = tree.search(dist, numNeighbours, INFINITY);

    // Remove the closest one if asked (this is needed when this example itself is
    // in the tree
    if (removeOne) {
        // Check that it really did have zero distance
        ExcAssertEqual(exNeighbours[0].first, 0.0);

        ExcAssertGreaterEqual(exNeighbours.size(), 1);

        exNeighbours.erase(exNeighbours.begin(), exNeighbours.begin() + 1);
    }

    // Sort by index number
    sort_on_second_ascending(exNeighbours);

    // Extract into separate vectors
    vector<int> indexes(exNeighbours.size());
    distribution<float> distances(exNeighbours.size());

    for (unsigned i = 0;  i < exNeighbours.size();  ++i)
        std::tie(distances[i], indexes[i]) = exNeighbours[i];
        
    // Now calculate the perplexity
    std::tie(result.probs, std::ignore)
        = binary_search_perplexity(distances, perplexity, -1, tolerance);

    // put it back in the node
    result.indexes = std::move(indexes);
    
    return result;
}

std::vector<TsneSparseProbs>
symmetrize(const std::vector<TsneSparseProbs> & input)
{
    // 1.  Convert to a sparse matrix format, and accumulate
    std::vector<ML::Lightweight_Hash<int, float> > probs(input.size());
    
    for (unsigned j = 0;  j < input.size();  ++j) {
        const TsneSparseProbs & p = input[j];
        for (unsigned i = 0;  i < p.indexes.size();  ++i) {
            // +1 is to avoid inserting 0 into a lightweight hash
            probs[p.indexes[i]][j + 1] += p.probs[i];
            probs[j][p.indexes[i] + 1] += p.probs[i];
        }
    }
    
    // 2.  Convert back to TsneSparseProbs, normalizing as we go
    std::vector<TsneSparseProbs> result(input.size());

    for (unsigned j = 0;  j < input.size();  ++j) {
        std::vector<std::pair<int, float> >
            sorted(probs[j].begin(), probs[j].end());
        std::sort(sorted.begin(), sorted.end());

        for (auto & s: sorted) {
            result[j].indexes.push_back(s.first - 1);
            result[j].probs.push_back(s.second / (2.0 * input.size()));
        }
    }

    return result;
}

boost::multi_array<float, 2>
tsneApproxFromCoords(const boost::multi_array<float, 2> & coords,
                     int num_dims,
                     const TSNE_Params & params,
                     const TSNE_Callback & callback,
                     std::unique_ptr<VantagePointTree> * treeOut,
                     std::unique_ptr<Quadtree> * qtreeOut)
{
    std::vector<TsneSparseProbs> neighbours
        = sparseProbsFromCoords(coords, params.numNeighbours,
                                params.perplexity, params.tolerance, treeOut);

    std::vector<TsneSparseProbs> symmetricNeighbours
        = symmetrize(neighbours);
    
    boost::multi_array<float, 2> embedding
        = tsneApproxFromSparse(symmetricNeighbours, num_dims, params, callback, qtreeOut);
    
    return embedding;
}


struct CalcRepContext {
    CalcRepContext(const ML::distribution<float> & y,
                   double * FrepZ,
                   double & exampleZ,
                   int & nodesTouched,
                   int nd,
                   bool exact,
                   const std::function<void (const QuadtreeNode & node,
                                             double qCellZ, const std::vector<int> & poi)> & onNode,
                   const std::function<const QCoord & (int)> & getPointCoord)
        : y(y), FrepZ(FrepZ), exampleZ(exampleZ), nodesTouched(nodesTouched),
          nd(nd), exact(exact), onNode(onNode), getPointCoord(getPointCoord)
    {
    }


    const ML::distribution<float> & y;
    double * FrepZ;
    double & exampleZ;
    int & nodesTouched;
    int nd;
    bool exact;
    const std::function<void (const QuadtreeNode & node,
                              double qCellZ, const std::vector<int> & poi)> & onNode;

    /// Used to get the coordinate of a point of interest passed in pointsInside
    const std::function<const QCoord & (int)> & getPointCoord;

    std::vector<int> NO_POINTS;

    void calc(const QuadtreeNode & node,
              int depth,
              bool inside,
              const std::vector<int> & pointsInside)
    {

        float com[nd];

        ++nodesTouched;

        float distSq = 0.0f;

        int effectiveNumChildren = node.numChildren - inside;

        if (effectiveNumChildren == 0)
            return;

        float ncr = node.recipNumChildren[inside];

        if (nd == 2) {
            com[0] = ((node.centerOfMass[0] - inside*y[0]) * ncr) - y[0];
            com[1] = ((node.centerOfMass[1] - inside*y[1]) * ncr) - y[1];
            distSq = com[0] * com[0] + com[1] * com[1];
        }
        else {
            for (unsigned i = 0;  i < nd;  ++i) {
                com[i] = ((node.centerOfMass[i] - inside*y[i]) * ncr) - y[i];
                distSq += com[i] * com[i];
            }
        }

        if (node.type == QuadtreeNode::TERMINAL
            || effectiveNumChildren == 1
            || (node.diag < 0.6 * sqrtf(distSq) && !exact)) {

            float qCellZ = 1.0f / (1.0f + distSq);
            if (distSq == 0.0) {
                cerr << "DISTANCE OF ZERO" << endl;
                cerr << "effectiveNumChildren = "
                     << effectiveNumChildren << endl;
                cerr << "node.numChildren = " << node.numChildren
                     << endl;
                cerr << "inside = " << inside << endl;
                cerr << "node.mins = " << node.mins << endl;
                cerr << "node.maxs = " << node.maxs << endl;
                cerr << "node.center = " << node.center << endl;
                cerr << "node.child = " << node.child << endl;
                cerr << "point = " << y << endl;
            }


            exampleZ += effectiveNumChildren * qCellZ;

            for (unsigned i = 0;  i < nd;  ++i) {
                FrepZ[i] += effectiveNumChildren * com[i] * qCellZ * qCellZ;
            }

            if (onNode) {
                onNode(node, qCellZ, pointsInside);
            }

            return;
        }
        
        // If we have points we are bringing along for the ride, then split them
        // by quadrant.
        if (!pointsInside.empty()) {
            std::vector<int> quadrantPoints[1 << nd];
            for (int p: pointsInside) {
                int quad = node.quadrant(getPointCoord(p));
                quadrantPoints[quad].push_back(p);
            }

            int quad = -1;
            if (inside)
                quad = node.quadrant(y);
            for (auto & q: node.quadrants) {
                calc(*q.second, depth + 1, q.first == quad, quadrantPoints[q.first]);
            }
        }
        else {
            int quad = -1;
            if (inside)
                quad = node.quadrant(y);
            for (auto & q: node.quadrants) {
                calc(*q.second, depth + 1, q.first == quad, NO_POINTS);
            }
        }
    }
};

// Used to traverse the quadtree for the Ys
void calcRep(const QuadtreeNode & node,
             int depth,
             bool inside,
             const ML::distribution<float> & y,
             double * FrepZ,
             double & exampleZ,
             int & nodesTouched,
             int nd,
             bool exact,
             const std::function<void (const QuadtreeNode & node,
                                       double qCellZ, const std::vector<int> & poi)> & onNode,
             const std::vector<int> & pointsOfInterest,
             const std::function<const QCoord & (int)> & getPointCoord)
{
    CalcRepContext context(y, FrepZ, exampleZ, nodesTouched, nd, exact, onNode, getPointCoord);
    context.calc(node, depth, inside, pointsOfInterest);
}

boost::multi_array<float, 2>
tsneApproxFromSparse(const std::vector<TsneSparseProbs> & exampleNeighbours,
                     int num_dims,
                     const TSNE_Params & params,
                     const TSNE_Callback & callback,
                     std::unique_ptr<Quadtree> * qtreeOut)
{
    // See van der Marten, 2013 http://arxiv.org/pdf/1301.3342.pdf
    // Barnes-Hut-SNE

    int nx = exampleNeighbours.size();
    int nd = num_dims;

    boost::multi_array<float, 2> Y = tsne_init(nx, nd, params.randomSeed);

    // Do we force calculations to be made exactly?
    bool forceExactSolution = false;
    //forceExactSolution = true;

    // Z * Frep
    boost::multi_array<double, 2> FrepZ(boost::extents[nx][nd]);

    // Y delta
    boost::multi_array<float, 2> dY(boost::extents[nx][nd]);

    // Last change in Y; so that we can see if we're going in the same dir
    boost::multi_array<float, 2> iY(boost::extents[nx][nd]);

    // Per-variable factors to multiply the gradient by to improve convergence
    boost::multi_array<float, 2> gains(boost::extents[nx][nd]);
    std::fill(gains.data(), gains.data() + gains.num_elements(), 1.0f);

    boost::multi_array<double, 2> FattrApprox(boost::extents[nx][nd]);
    boost::multi_array<double, 2> FrepApprox(boost::extents[nx][nd]);

    boost::multi_array<float, 2> lastNormalizedY(boost::extents[nx][nd]);

    double cost = INFINITY;
    double last_cost = INFINITY;
    
    if (callback
        && !callback(-1, cost, "init")) return Y;

    Timer timer;

    // As described in Hinton et al, start off with a total probability of 4, before moving
    // back to 1 after 100 iterations.
    float pFactor = 4.0;

    //cerr << "exampleNeighbours[0].indexes = " << exampleNeighbours[0].indexes << endl;
    //cerr << "exampleNeighbours[0].probs = " << exampleNeighbours[0].probs << endl;
    //cerr << "exampleNeighbours[0].probs.total() = " << exampleNeighbours[0].probs.total() << endl;
    //cerr << "exampleNeighbours[0].probs.min() = " << exampleNeighbours[0].probs.min() << endl;
    //cerr << "exampleNeighbours[0].probs.max() = " << exampleNeighbours[0].probs.max() << endl;

    //cerr << "sump0 = " << exampleNeighbours[0].probs.total() * pFactor << endl;
    //cerr << "sump1 = " << exampleNeighbours[1].probs.total() * pFactor << endl;
    
    for (int iter = 0;  iter < params.max_iter;  ++iter) {

        //cerr << "iter " << iter << endl;

        // Find the bounding box for the quadtree
        ML::distribution<float> mins(nd), maxs(nd);

        for (unsigned j = 0;  j < nx;  ++j) {
            ML::distribution<float> y(nd);
            for (unsigned i = 0;  i < nd;  ++i)
                y[i] = Y[j][i];
            
            if (j == 0)
                mins = maxs = y;
            else {
                y.min_max(mins, maxs);
            }
        }

        // Create the quadtree for this iteration
        QCoord minc(mins.begin(), mins.end()), maxc(maxs.begin(), maxs.end());

        // Bounding boxes are open ended on the max side, so move to the next float
        for (float & c: maxc) {
            c = nextafterf(c, (float)INFINITY);
        }

        Quadtree qtree(minc, maxc);

        // Insert the values into the quadtree
        for (unsigned i = 0;  i < nx;  ++i) {
            QCoord coord(nd);
            for (unsigned j = 0;  j < nd;  ++j) {
                coord[j] = Y[i][j];
            }

            qtree.insert(coord);
        }
        
        int numNodes JML_UNUSED = qtree.root->finish();

        //cerr << "points are in " << numNodes << " nodes" << endl;

        // Create a new coordinate for each neighbour
        std::vector<QCoord> pointCoords(nx);

        for (unsigned i = 0;  i < nx;  ++i) {
            pointCoords[i] = QCoord(&Y[i][0], &Y[i][0] + nd);
        }

        // This accumulates the sum_j p[x][j] log Z*q[x][j] for each example.  From this and
        // Z, we can calculate the cost of each example.  Only relevant if calcC is true.
        double exampleCFactor[nx];
        std::fill(exampleCFactor, exampleCFactor  +nx, 0.0);

        // Do we calculate the cost?
        bool calcC = iter < 10 || (iter + 1) % 100 == 0 || iter == params.max_iter - 1;

        // Approximation for Z, accumulated here
        ML::Spinlock Zmutex;
        std::vector<double> ZApproxValues;
        ZApproxValues.reserve(nx);

        auto calcExample = [&] (int x)
            {
                // Clear the updates
                for (unsigned i = 0;  i < nd;  ++i) {
                    dY[x][i] = 0.0;
                    FrepZ[x][i] = 0.0;
                    FattrApprox[x][i] = 0.0;
                    FrepApprox[x][i] = 0.0;
                }

                const TsneSparseProbs & neighbours = exampleNeighbours[x];
                
                distribution<float> y(nd);
                for (unsigned i = 0;  i < nd;  ++i)
                    y[i] = Y[x][i];

                // For each neighbour, calculate the attractive force.  The
                // others are defined as zero.
                for (unsigned q = 0;  q < neighbours.indexes.size();  ++q) {
                    
                    unsigned j = neighbours.indexes[q];
                    ExcAssertNotEqual(j, x);

                    double D = 0.0;
                    if (nd == 2) {
                        float d0 = y[0] - Y[j][0];
                        float d1 = y[1] - Y[j][1];
                        D = d0 * d0 + d1 * d1;
                    } else {
                        for (unsigned i = 0;  i < nd;  ++i) {
                            D += (y[i] - Y[j][i]) * (y[i] - Y[j][i]);
                        }
                    }

                    //if (x == 0 && j == 1) {
                    //    cerr << "D[0][1] approx = " << D << " prob "
                    //         << neighbours.probs[q] << endl;
                    //}

                    // Note that 1/(1 + D[j]) == Q[j] * Z
                    // See van der Marten, 2013 http://arxiv.org/pdf/1301.3342.pdf
                    // Barnes-Hut-SNE

                    double factorAttr = pFactor * neighbours.probs[q] / (1.0 + D);

                    if (nd == 2) {
                        float dYj0 = y[0] - Y[j][0];
                        float dYj1 = y[1] - Y[j][1];
                        FattrApprox[x][0] += dYj0 * factorAttr;
                        FattrApprox[x][1] += dYj1 * factorAttr;
                    }
                    else {
                        for (unsigned i = 0;  i < nd;  ++i) {
                            double dYji = y[i] - Y[j][i];
                            FattrApprox[x][i] += dYji * factorAttr;
                        }
                    }
                }

                // Working storage for onNode
                ML::distribution<double> com(nd);

                int nodesTouched = 0;

                double exampleZ = 0.0;

                //bool doingTest = false;

                bool exact = forceExactSolution;

                int poiDone = 0;

                auto onNode = [&] (const QuadtreeNode & node,
                                   double qCellZ,
                                   const std::vector<int> & pointsOfInterest)
                {
                    // If we want to calculate C, we store the log of
                    // the cell's Q * Z for each point of interest so that
                    // we can calculate the cost later.

                    // Note that sum_j p[j] log (Zq[j])
                    //         = sum_j p[j] log Z + sum_j p[j] log q[j]

                    double logqCellZ = log(qCellZ);
                    for (unsigned p: pointsOfInterest) {
                        exampleCFactor[x] += pFactor * neighbours.probs[p] * logqCellZ;
                    }

                    poiDone += pointsOfInterest.size();
                };

                auto getPointCoord = [&] (int point) -> const QCoord &
                {
                    return pointCoords.at(neighbours.indexes.at(point));
                };

                if (calcC) {
                    // Bring along the points of interest for the ride
                    vector<int> pointsOfInterest;
                    pointsOfInterest.reserve(neighbours.indexes.size());
                    for (unsigned i = 0;  i < neighbours.indexes.size();  ++i)
                        pointsOfInterest.push_back(i);

                    calcRep(*qtree.root, 0, true /* inside */,
                            y, &FrepZ[x][0], exampleZ, nodesTouched, nd, exact,
                            onNode, pointsOfInterest, getPointCoord);
                    ExcAssertEqual(poiDone, neighbours.indexes.size());
                    //if (!isfinite(exampleCFactor[x]))
                    //    cerr << "x = " << x << " factor " << exampleCFactor[x] << endl;
                    ExcAssert(isfinite(exampleCFactor[x]));
                } else {
                    calcRep(*qtree.root, 0, true /* inside */,
                            y, &FrepZ[x][0], exampleZ, nodesTouched, nd, exact,
                            nullptr, {}, nullptr);
                }

                {
                    std::unique_lock<ML::Spinlock> guard(Zmutex);
                    ZApproxValues.push_back(exampleZ);
                }

                //if (x == 1026)
                //    cerr << "touched " << nodesTouched << " of " << numNodes << " nodes"
                //         << endl;
            };

#if 1
        int totalThreads = 4;

        auto doThread = [&] (int n)
            {
                int perThread = nx / totalThreads;
                int start = n * perThread;
                int end = start + perThread;
                if (n == totalThreads)
                    end = nx;

                for (unsigned x = start;  x < end;  ++x)
                    calcExample(x);

                //for (unsigned x = n;  x < nx;  x += totalThreads) {
                //    calcExample(x);
                //}
            };

        ML::run_in_parallel(0, totalThreads, doThread);
        //ML::run_in_parallel_blocked(0, nx, calcExample);
#else
        // Each example proceeds more or less independently
        for (unsigned x = 0;  x < nx;  ++x) {
            calcExample(x);
        }
#endif

        // Sort from smallest to largest to accumulate.  This minimises
        // rounding errors and makes the result independent of the order
        // in which threads finish.
        std::sort(ZApproxValues.begin(), ZApproxValues.end());
        double ZApprox = std::accumulate(ZApproxValues.begin(),
                                         ZApproxValues.end(),
                                         0.0);

        double Zrecip = 1.0 / ZApprox;

        for (unsigned x = 0;  x < nx;  ++x) {
            for (unsigned i = 0;  i < nd;  ++i) {
                FrepApprox[x][i] = FrepZ[x][i] * Zrecip;
            }
        }

        double Capprox = 0.0;
        if (calcC) {
            //double logZ = log(ZApprox);

            // For a given example x,
            // C[x] = sum_j P[x][j] log P[x][j] - sum_j P[x][j] log q[x][j]
            //      = sum_j P[x][j] log P[x][j] - sum_j P[x][j] log Zq[j][j] + sum_j P[x][j] log Z
            //      = sum_j P[x][j] log Z P[x][j] - exampleCFactor[x]
            for (unsigned x = 0;  x < nx;  ++x) {

                const TsneSparseProbs & neighbours = exampleNeighbours[x];

                double Cexample = -exampleCFactor[x];

                for (auto & p: neighbours.probs) {
                    Cexample += pFactor * p * logf(pFactor * p * ZApprox);
                }

                Capprox += Cexample;
            }
        }

#if 0  // exact calculations for verification        
        double Z = 0.0, C = 0.0;

        boost::multi_array<float, 2> QZ(boost::extents[nx][nx]);
        boost::multi_array<double, 2> Fattr(boost::extents[nx][nd]);
        boost::multi_array<double, 2> Frep(boost::extents[nx][nd]);
        
        for (unsigned x = 0;  x < nx;  ++x) {

            distribution<float> y(nd);
            for (unsigned i = 0;  i < nd;  ++i) {
                y[i] = Y[x][i];
                Fattr[x][i] = 0.0;
            }

            for (unsigned j = 0;  j < nx;  ++j) {
                if (j == x)
                    continue;

                //if (x == 0 && j == 1) {
                //    cerr << "D[0][1] real   = " << D << " prob "
                //         << P[x][j] << endl;
                //}

            }

            for (unsigned j = 0;  j < nx;  ++j) {
                if (j == x)
                    continue;

                // Distances, used to calculate Q and Z
                double D = 0.0;
                if (nd == 2) {
                    float d0 = y[0] - Y[j][0];
                    float d1 = y[1] - Y[j][1];
                    D = d0 * d0 + d1 * d1;
                } else {
                    for (unsigned i = 0;  i < nd;  ++i) {
                        D += (y[i] - Y[j][i]) * (y[i] - Y[j][i]);
                    }
                }

                QZ[x][j] = 1.0 / (1.0 + D);
                Z += QZ[x][j];
            }

            const TsneSparseProbs & neighbours = exampleNeighbours[x];
            
            // For each neighbour, calculate the attractive force.  The
            // others are defined as zero.
            for (unsigned q = 0;  q < neighbours.indexes.size();  ++q) {
                    
                unsigned j = neighbours.indexes[q];
                ExcAssertNotEqual(j, x);

                double factorAttr = pFactor * neighbours.probs[q] * QZ[x][j];

                if (nd == 2) {
                    double dYj0 = y[0] - Y[j][0];
                    double dYj1 = y[1] - Y[j][1];
                    Fattr[x][0] += dYj0 * factorAttr;
                    Fattr[x][1] += dYj1 * factorAttr;
                }
                else {
                    for (unsigned i = 0;  i < nd;  ++i) {
                        double dYji = y[i] - Y[j][i];
                        Fattr[x][i] += dYji * factorAttr;
                    }
                }

            }
        }

        //cerr << "ZApprox = " << ZApprox << " Z = " << Z << endl;


        for (unsigned x = 0;  x < nx;  ++x) {
            distribution<float> y(nd);
            for (unsigned i = 0;  i < nd;  ++i) {
                y[i] = Y[x][i];
                Frep[x][i] = 0;
            }

            for (unsigned j = 0;  j < nx;  ++j) {
                if (j == x)
                    continue;

                double Qxj = QZ[x][j] / Z;

                //Qxj = std::max<double>(params.min_prob, Qxj);

                // Repulsive force
                float factorRep = Qxj * Z * Qxj;

                if (nd == 2) {
                    float dYj0 = y[0] - Y[j][0];
                    float dYj1 = y[1] - Y[j][1];
                    Frep[x][0] -= dYj0 * factorRep;
                    Frep[x][1] -= dYj1 * factorRep;
                }
                else {
                    for (unsigned i = 0;  i < nd;  ++i) {
                        double dYji = y[i] - Y[j][i];
                        Frep[x][i] -= dYji * factorRep;
                    }
                }

            }

            const TsneSparseProbs & neighbours = exampleNeighbours[x];
            
            // For each neighbour, calculate the attractive force.  The
            // others are defined as zero.
            for (unsigned q = 0;  q < neighbours.indexes.size();  ++q) {
                    
                unsigned j = neighbours.indexes[q];
                ExcAssertNotEqual(j, x);

                double Qxj = QZ[x][j] / Z;

                C += pFactor * neighbours.probs[q] * logf(pFactor * neighbours.probs[q] / Qxj);
            }
        }

        cerr << "Capprox = " << Capprox << " C = " << C << endl;
#endif

        float maxAbsDy = 0.0;
        float maxAbsY = 0.0;

        for (unsigned x = 0;  x < nx;  ++x) {
            for (unsigned i = 0;  i < nd;  ++i) {
                //dY[x][i] = 4.0 * (Fattr[x][i] + Frep[x][i]);
                //dY[x][i] = 4.0 * (FattrApprox[x][i] + Frep[x][i]);
                //dY[x][i] = 4.0 * (Fattr[x][i] + FrepApprox[x][i]);
                dY[x][i] = 4.0 * (FattrApprox[x][i] + FrepApprox[x][i]);

                maxAbsDy = std::max(maxAbsDy, fabs(dY[x][i]));
                maxAbsY = std::max(maxAbsY, Y[x][i]);

#if 0
                if (x < 5) {
                    cerr << "Fattr[" << x << "][" << i << "]: approx "
                         << FattrApprox[x][i] << " real " << Fattr[x][i]
                         << endl;
                    cerr << "Frep[" << x << "][" << i << "]: approx "
                         << FrepApprox[x][i] << " real " << Frep[x][i]
                         << endl;
                }

                if (x < 5) {
                    cerr << "dY[" << x << "][" << i << "]: approx "
                         << 4.0 * (FattrApprox[x][i] + FrepApprox[x][i])
                         << " real " << 4.0 * (Fattr[x][i] + Frep[x][i])
                         << endl;
                    cerr << "Y = " << Y[x][i] << endl;
                }
#endif
            }

        }

#if 0
        cerr << "C = " << Capprox << endl;

        for (unsigned x = 0;  x < 5;  ++x) {
            cerr << "P[" << x << "][0..5] = "
                 << pFactor * exampleNeighbours[x].probs[0]
                 << " " << pFactor * exampleNeighbours[x].probs[1]
                 << " " << pFactor * exampleNeighbours[x].probs[2]
                 << " " << pFactor * exampleNeighbours[x].probs[3]
                 << " " << pFactor * exampleNeighbours[x].probs[4]
                 << endl;

            cerr << "P[" << x << "][0..5] = "
                 << exampleNeighbours[x].indexes[0]
                 << " " << exampleNeighbours[x].indexes[1]
                 << " " << exampleNeighbours[x].indexes[2]
                 << " " << exampleNeighbours[x].indexes[3]
                 << " " << exampleNeighbours[x].indexes[4]
                 << endl;

                for (unsigned i = 0;  i < nd;  ++i) {
                    cerr << "dY[" << x << "][" << i << "]: real " << dY[x][i]
                         << endl;
                    cerr << "Y = " << Y[x][i] << endl;
                }
        }

        return Y;
#endif

        double cost2 = Capprox;
        if (calcC) {
            //cerr << "Cost approx " << Capprox << " real " << C << endl;

            last_cost = cost;
            cost = cost2;

            if (isfinite(cost) && cost == last_cost) {
                // converged
                break;
            }
        }

        /*********************************************************************/
        // Update

        float momentum = (iter < 20
                          ? params.initial_momentum
                          : params.final_momentum);

        tsne_update(Y, dY, iY, gains, iter == 0, momentum, params.eta,
                    params.min_gain);

        if (callback
            && !callback(iter, cost, "update")) return Y;


        /*********************************************************************/
        // Recenter about the origin

        recenter_about_origin(Y);

        if (callback
            && !callback(iter, cost, "recenter")) return Y;

        if ((iter + 1) % 100 == 0 || iter == params.max_iter - 1) {
            cerr << format("iteration %4d cost %6.3f  ",
                           iter + 1, cost)
                 << timer.elapsed() << endl;
            timer.restart();
        }
        
        float maxAbsCoord[nd];
        std::fill(maxAbsCoord, maxAbsCoord + nd, 0.0);

        boost::multi_array<float, 2> normalizedY(boost::extents[nx][nd]);
        for (unsigned x = 0;  x < nx;  ++x) {
            for (unsigned i = 0;  i < nd;  ++i) {
                maxAbsCoord[i] = std::max(maxAbsCoord[i], fabs(Y[x][i]));
            }
        }

        float maxCoordChange = 0.0;

        for (unsigned x = 0;  x < nx;  ++x) {
            for (unsigned i = 0;  i < nd;  ++i) {
                normalizedY[x][i] = Y[x][i] / maxAbsCoord[i];
                maxCoordChange = std::max(maxCoordChange, fabs(normalizedY[x][i] - lastNormalizedY[x][i]));
            }
        }

        lastNormalizedY = normalizedY;

        //cerr << "maxAbsDy = " << maxAbsDy << " maxAbsY = " << maxAbsY
        //     << " ratio " << 100.0 * maxAbsDy / maxAbsY
        //     << " maxCoordChange = " << maxCoordChange << endl;

        if (maxCoordChange < 0.001 && iter > 200)
            return Y;
            
        // Stop lying about P values if we're finished 100 iterations
        if (iter == 100) {
            pFactor /= 4.0;
        }
    }

    return Y;
}

ML::distribution<float>
retsne(const ML::distribution<float> & probs_,
       const boost::multi_array<float, 2> & prevOutput,
       const TSNE_Params & params)
{
    int nx = prevOutput.shape()[0];
    int nd = prevOutput.shape()[1];

    ML::distribution<float> mins(nd), maxs(nd);
    std::vector<ML::distribution<float> > Y(nx, ML::distribution<float>(nd));

    for (unsigned i = 0;  i < nx;  ++i) {
        for (unsigned j = 0;  j < nd;  ++j) {
            Y[i][j] = prevOutput[i][j];
        }
        if (i == 0)
            mins = maxs = Y[i];
        else {
            Y[i].min_max(mins, maxs);
        }
        //cerr << "input " << i << " is " << Y[i] << " with prob " << probs[i] << endl;
    }

    QCoord minc(mins.begin(), mins.end()), maxc(maxs.begin(), maxs.end());

    // Bounding boxes are open ended on the max side, so move to the next float
    for (float & c: maxc) {
        c = nextafterf(c, (float)INFINITY);
    }

    Quadtree qtree(minc, maxc);

    for (unsigned i = 0;  i < nx;  ++i) {
        QCoord coord(nd);
        for (unsigned j = 0;  j < nd;  ++j) {
            coord[j] = prevOutput[i][j];
        }

        qtree.insert(coord);
    }

    int numNodes = qtree.root->finish(0);

    //cerr << "quadtree had " << numNodes << " nodes for " << qtree.root->numChildren
    //     << " children" << endl;

    ExcAssertEqual(qtree.root->numChildren, nx);

    distribution<float> probs = probs_;

    //cerr << "input perplexity is " << perplexity(probs_) << endl;
    //cerr << "input probs total " << probs_.total() << " min " << probs.min()
    //     << " max " << probs.max() << endl;

    // 1.  Calculate the probabilities.  We can start at the origin

    //cerr << "nd = " << nd << " nx = " << nx << endl;


    probs.normalize();

    std::vector<std::pair<float, int> > probsSorted;
    for (unsigned i = 0;  i < probs.size();  ++i) {
        probsSorted.emplace_back(probs[i], i);
    }

    std::sort(probsSorted.begin(), probsSorted.end());
    std::reverse(probsSorted.begin(), probsSorted.end());

    //cerr << "probsSorted = " << probsSorted << endl;

    double probsTotal = 0.0;
    int numNeeded = -1;
    for (unsigned i = 0;  i < probsSorted.size();  ++i) {
        probsTotal += probsSorted[i].first;
        if (probsTotal > 0.99) {
            numNeeded = i;
            break;
        }
    }

    // Start off at the Y of the point with the highest probability, to get faster
    // convergance
    ML::distribution<float> y(nd);

    y = Y[probsSorted[0].second];

    // Restricted subset of neighbours used to approximate attractive
    // force.
    vector<int> neighbours;
    ML::distribution<float> neighbourProbs;
    for (unsigned i = 0;  i < numNeeded && i < 60;  ++i) {
        neighbours.push_back(probsSorted[i].second);
        neighbourProbs.push_back(probsSorted[i].first);
    }

    neighbourProbs.normalize();

    // Now for each node, figure out which of the neighbour points are there
    
    std::vector<ML::compact_vector<int, 3> > nodeNeighbours(numNodes);

    // Create a new coordinate for each neighbour
    std::vector<QCoord> neighbourCoords(neighbours.size());

    for (unsigned i = 0;  i < neighbours.size();  ++i) {
        neighbourCoords[i] = QCoord(&Y[neighbours[i]][0], &Y[neighbours[i]][0] + nd);
    }

    // Now traverse the quadtree, splitting up the list of neighbours so that we
    // have a list of them at each node
    std::function<void (QuadtreeNode & node, const std::vector<int> &) > distributeNeighbours
        = [&] (QuadtreeNode & node,
               const std::vector<int> & neighbours)
        {
            nodeNeighbours[node.nodeNumber].insert
                (nodeNeighbours[node.nodeNumber].begin(),
                 neighbours.begin(), neighbours.end());
            
            if (node.type == QuadtreeNode::TERMINAL) {
                ExcAssertEqual(neighbours.size(), 1);
                ExcAssertEqual(neighbourCoords[neighbours[0]],
                               node.centerOfMass);
            }
            else {
                std::map<int, std::vector<int> > quadrants;

                for (auto & n: neighbours) {
                    quadrants[node.quadrant(node.center, neighbourCoords[n])].push_back(n);
                }

                for (auto & q: quadrants) {
                    auto it = node.quadrants.find(q.first);
                    ExcAssert(it != node.quadrants.end());
                    distributeNeighbours(*it->second, q.second);
                }
            }
        };

    // Distribute all neighbours across the quadtree
    vector<int> iota;
    for (unsigned i = 0;  i < neighbours.size();  ++i)
        iota.push_back(i);

    distributeNeighbours(*qtree.root, iota);

    //cerr << "done distributing" << endl;

    //y = {0.0, 0.0};
    //y = { -95.387, -55.4707 };

    double lastC = INFINITY;

    float D[nx], Q[nx];

    for (unsigned iter = 0;  iter < 1000;  ++iter) {

        // Y gradients
        double dy[nd];
        std::fill(dy, dy + nd, 0.0);

        // Y gradients
        double Fattr[nd];
        std::fill(Fattr, Fattr + nd, 0.0);

        // Approximate solution to the repulsive force
        bool calcC = iter % 20 == 0;
        double C = 0.0;

        double ZApprox = 0.0;
        double FrepZApprox[nd];
        std::fill(FrepZApprox, FrepZApprox + nd, 0.0);

        ML::distribution<float> com(nd);

        float logqZ[neighbours.size()];
        std::fill(logqZ, logqZ + neighbours.size(), std::numeric_limits<float>::quiet_NaN());

        auto onNode = [&] (const QuadtreeNode & node, int depth)
            {
                double dist;
                
                if (nd == 2) {
                    float ncr = 1.0 / node.numChildren;
                    com[0] = (node.centerOfMass[0] * ncr) - y[0];
                    com[1] = (node.centerOfMass[1] * ncr) - y[1];
                    dist = sqrt(com[0] * com[0] + com[1] * com[1]);
                }
                else {

                    // 1.  Calculate the distance between this y point and the
                    //     node center
                    std::copy(node.centerOfMass.begin(),
                              node.centerOfMass.end(),
                              com.begin());

                    // Normalized by number of children
                    com *= 1.0 / node.numChildren;

                    // Turn into a distance from y
                    com -= y;

                    dist = com.two_norm();
                }

                double diag = node.diagonalLength();
                double ratio = diag / dist;

                //cerr << "dist = " << dist << " diag = " << diag
                //     << " ratio = " << ratio << endl;

                if (node.numChildren == 1 || ratio < 0.4) {
                    // Stop here

                    //if (node.numChildren > 1)
                    //    cerr << "early stop with " << node.numChildren
                    //         << " children" << endl;

                    double qCellZ = node.numChildren / (1.0 + dist * dist);

                    ZApprox += qCellZ;

                    for (unsigned i = 0;  i < nd;  ++i) {
                        FrepZApprox[i] += com[i] * qCellZ * qCellZ;
                    }

                    // If we want to calculate C, we store the log of
                    // the cell's Q * Z for each point of interest so that
                    // we can calculate the cost later.
                    if (calcC) {
                        auto & neighbours = nodeNeighbours[node.nodeNumber];
                        if (!neighbours.empty()) {
                            float logqCellZ = logf(qCellZ);
                            for (int neighbour: neighbours) {
                                logqZ[neighbour] = logqCellZ;
                            }
                        }
                    }
                    return false;
                }

                return true;  // continue recursing
            };

        qtree.root->walk(onNode);

        double FrepApprox[nd];
        for (unsigned i = 0;  i < nd;  ++i) {
            FrepApprox[i] = FrepZApprox[i] / ZApprox;
        }

        double FattrApprox[nd];
        std::fill(FattrApprox, FattrApprox + nd, 0.0);

        for (unsigned q = 0;  q < neighbours.size();  ++q) {
            unsigned j = neighbours[q];

            double D = 0.0;
            if (nd == 2) {
                float d0 = y[0] - Y[j][0];
                float d1 = y[1] - Y[j][1];
                D = d0 * d0 + d1 * d1;
            } else {
                for (unsigned i = 0;  i < nd;  ++i) {
                    D += (y[i] - Y[j][i]) * (y[i] - Y[j][i]);
                }
            }

            // Note that 1/(1 + D[j]) == Q[j] * Z
            // See van der Marten, 2013 http://arxiv.org/pdf/1301.3342.pdf
            // Barnes-Hut-SNE

            float factorAttr = neighbourProbs[q] / (1.0f + D);

            if (nd == 2) {
                float dYj0 = y[0] - Y[j][0];
                float dYj1 = y[1] - Y[j][1];
                FattrApprox[0] += dYj0 * factorAttr;
                FattrApprox[1] += dYj1 * factorAttr;
            }
            else {
                for (unsigned i = 0;  i < nd;  ++i) {
                    double dYji = y[i] - Y[j][i];
                    FattrApprox[i] += dYji * factorAttr;
                }
            }

            //if (calcC) {
            //    C += probs[j] * (log(probs[j]) - logQ(probs[j]));
            //}
        }

        double Capprox = 0.0;
        if (calcC) {
            float logZ = log(ZApprox);
            for (unsigned q = 0;  q < neighbours.size();  ++q) {
                // log (Z * qj) = logZ + log qj, so log qj = log (Z * qj) - log Z

                float logqj = logqZ[q] - logZ;
                Capprox += neighbourProbs[q] * (logf(neighbourProbs[q]) - logqj);
            }
        }

        if (calcC && false) {
            double Z = 0.0;

            // Square of pythagorean distances of this point from each other
            // point in low dimensional space
            for (unsigned j = 0;  j < nx;  ++j) {
                D[j] = 0.0;
                if (nd == 2) {
                    float d0 = y[0] - Y[j][0];
                    float d1 = y[1] - Y[j][1];
                    D[j] = d0 * d0 + d1 * d1;
                } else {
                    for (unsigned i = 0;  i < nd;  ++i) {
                        D[j] += (y[i] - Y[j][i]) * (y[i] - Y[j][i]);
                    }
                }

                //auto dist = y - Y[j];
                //D[j] = dist.dotprod(dist);
                Q[j] = 1.0f / (1.0f + D[j]);
                Z += Q[j];
            }


            for (unsigned j = 0;  j < nx;  ++j) {
                Q[j] /= Z;
            }

            for (unsigned q = 0;  q < neighbours.size();  ++q) {
                float p = neighbourProbs[q];
                C += p * logf(p / Q[neighbours[q]]);
            }

            cerr << "C = " << C << " Capprox = " << Capprox << endl;
        }

        C = Capprox;

        for (unsigned i = 0;  i < nd;  ++i) {
            dy[i] += FrepApprox[i];
        }

        for (unsigned i = 0;  i < nd;  ++i) {
            dy[i] += FattrApprox[i];
        }

        //dy *= 2.0;

        //cerr << "iter " << iter << " C = " << C << " y = " << y
        //     << " dY = " << dy << endl;
            
        if (calcC) {
            if (fabs(C - lastC) < 0.00001) {
                //cerr << "converged after " << iter << " iterations" << endl;
                break;
            }
            lastC = C;
        }

        for (unsigned i = 0;  i < nd;  ++i)
            y[i] -= 100.0 * dy[i];

        //y -= 100.0 * dy;

        //cerr << "dy = " << dy << " dy_num = " << dY_num << " y now " << y << endl;
    }

    return y;
}



} // namespace ML
