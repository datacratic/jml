/** vantage_point_tree.h                                           -*- C++ -*-
    Jeremy Barnes, 18 November 2014
    Copyright (c) 2014 Datacratic Inc.  All rights reserved.

    Available under the BSD license, no attribution required.
*/

#pragma once

#include "jml/stats/distribution.h"
#include "jml/utils/exc_assert.h"
#include "jml/db/persistent.h"
#include "jml/utils/compact_vector.h"
#include <iostream>

namespace ML {

template<typename Item>
struct VantagePointTreeT {

    VantagePointTreeT(const std::vector<Item> & items, double radius,
                      std::unique_ptr<VantagePointTreeT> && inside,
                      std::unique_ptr<VantagePointTreeT> && outside)
        : items(items.begin(), items.end()),
          radius(radius),
          inside(std::move(inside)), outside(std::move(outside))
    {
    }

    VantagePointTreeT(ML::compact_vector<Item, 1> items, double radius,
                      std::unique_ptr<VantagePointTreeT> && inside,
                      std::unique_ptr<VantagePointTreeT> && outside)
        : items(std::move(items)),
          radius(radius),
          inside(std::move(inside)), outside(std::move(outside))
    {
    }

    /** Construct for a clump of items all of which are equidistant from the
        given item.
    */
    VantagePointTreeT(ML::compact_vector<Item, 1> items, double clumpRadius,
                      ML::compact_vector<Item, 1> clumpItems)
        : items(std::move(items)), radius(clumpRadius), clump(std::move(clumpItems))
    {
    }

    VantagePointTreeT(Item item, double radius,
                     std::unique_ptr<VantagePointTreeT> && inside,
                     std::unique_ptr<VantagePointTreeT> && outside)
        : items(1, item),
          radius(radius),
          inside(std::move(inside)), outside(std::move(outside))
    {
    }

    VantagePointTreeT(Item item)
        : items{item},
          radius(std::numeric_limits<float>::quiet_NaN())
    {
    }

    VantagePointTreeT(compact_vector<Item, 1> items)
        : items(std::move(items)),
          radius(std::numeric_limits<float>::quiet_NaN())
    {
    }

    VantagePointTreeT()
        : radius(INFINITY)
    {
    }

    ML::compact_vector<Item, 1> items; ///< All these have distance zero from each other
    double radius;  ///< Radius of the ball in which we choose inside versus outside

    /** Points that are in a clump with the same distance between the point and the
        pivot items.

        These are suspected or known to be distributed over the corners of a high
        dimensional hyper-cube.
    */
    ML::compact_vector<Item, 1> clump;
    
    /// Children that are inside the ball of the given radius on object
    std::unique_ptr<VantagePointTreeT> inside;

    /// Children that are outside the ball of given radius on the object
    std::unique_ptr<VantagePointTreeT> outside;

    static VantagePointTreeT *
    create(const std::vector<Item> & objectsToInsert,
           const std::function<float (Item, Item)> & distance)
    {
        auto distances = [&distance] (Item pivot,
                                      const std::vector<Item> & items2,
                                      int depth)
            {
                // Calculate distances to all children
                ML::distribution<float> distances(items2.size());

                for (unsigned i = 0;  i < items2.size();  ++i) {
                    distances[i] = distance(pivot, items2[i]);
                }

                return distances;
            };

        return createParallel(objectsToInsert, distances, 0);
    }

    static VantagePointTreeT *
    createParallel(const std::vector<Item> & objectsToInsert_,
                   const std::function<ML::distribution<float> (Item, const std::vector<Item> &, int)> & distance,
                   int depth = 0)
    {
        using namespace std;

        if (objectsToInsert_.empty())
            return nullptr;

        if (objectsToInsert_.size() == 1)
            return new VantagePointTreeT(objectsToInsert_[0]);

        // 1.  Choose a random object, in this case the middle one
        size_t index = objectsToInsert_.size() / 2;

        // Loop until we select a reasonable pivot element
        for (int attempt = 0;  ;  ++attempt) {

            vector<Item> objectsToInsert;
            objectsToInsert.reserve(objectsToInsert_.size());

            // Copy first half
            std::copy(objectsToInsert_.begin() + index,
                      objectsToInsert_.end(),
                      back_inserter(objectsToInsert));

            // Copy second half
            std::copy(objectsToInsert_.begin(),
                      objectsToInsert_.begin() + index,
                      back_inserter(objectsToInsert));

            ExcAssertEqual(objectsToInsert.size(), objectsToInsert_.size());

            Item pivot = objectsToInsert[0];

            // Calculate distances to all children
            ML::distribution<float> distances
                = distance(pivot, objectsToInsert, depth);

            ExcAssertEqual(distances.size(), objectsToInsert.size());

            //for (float d: distances)
            //    ExcAssert(isfinite(d));

            // Sort them
            std::vector<std::pair<float, Item> > sorted;
            sorted.reserve(objectsToInsert.size());
            for (unsigned i = 0;  i < objectsToInsert.size();  ++i) {
                if (!isfinite(distances[i])) {
                    using namespace std;
                    cerr << "distances = " << distances << endl;
                }
                ExcAssert(isfinite(distances[i]));
                ExcAssertGreaterEqual(distances[i], 0.0);
                sorted.emplace_back(distances[i], objectsToInsert[i]);
            }

            std::sort(sorted.begin(), sorted.end());

            // Find the first one that's not zero
            compact_vector<Item, 1> items;
            size_t firstNonZero = 0;
            while (firstNonZero < sorted.size() && sorted[firstNonZero].first == 0.0) {
                items.push_back(sorted[firstNonZero].second);
                ++firstNonZero;
            }

            if (items.size() == 0) {
                for (auto & s: sorted) {
                    cerr << s.first << "," << s.second << "  ";
                }
                cerr << endl;
                //cerr << "sorted = " << sorted << endl;
            }

            ExcAssertGreaterEqual(items.size(), 1);

            // If all have zero distance, just put them all in together
            if (firstNonZero == sorted.size()) {
                return new VantagePointTreeT(std::move(items));
            }

            // Get median distance, to use as a radius
            size_t splitPoint = firstNonZero + (distances.size() - firstNonZero) / 2;

            // If we have a split point that is the same as the minimum, increment
            // until it's not
            
            while (splitPoint < sorted.size() - 1
                   && sorted[splitPoint].first == sorted[firstNonZero].first)
                ++splitPoint;

            float radius = sorted.at(splitPoint).first;

            ExcAssertGreaterEqual(radius, sorted[firstNonZero].first);
            ExcAssertLessEqual(radius, sorted.back().first);

            if (attempt == 4 && distances[firstNonZero] == distances.back()) {
                // Completely homogeneous distances.  It's very likely that these
                // points are on the vertices of a high dimensional cube in the
                // metric space.

                // We don't make any attempt to separate them; we leave the work
                // until we query them

                compact_vector<Item, 1> clumpItems;
                for (unsigned i = firstNonZero;  i < distances.size();  ++i)
                    clumpItems.push_back(sorted[i].second);

                std::sort(clumpItems.begin(), clumpItems.end());
                
                cerr << "Completely homogeneous clump of "
                     << distances.size() - firstNonZero << " points" << endl;

                return new VantagePointTreeT(std::move(items), radius, std::move(clumpItems));
            }

        
            // Split into two subgroups
            std::vector<Item> insideObjects;
            std::vector<Item> outsideObjects;

            for (unsigned i = firstNonZero;  i < objectsToInsert.size();  ++i) {
                if (sorted[i].first <= radius)
                    insideObjects.push_back(sorted[i].second);
                else
                    outsideObjects.push_back(sorted[i].second);
            }
        
            // If we chose a bad split point, go back and choose a better one
            // we can do this up to 5 times.
            if (attempt != 4
                && outsideObjects.size() > 10
                && 1.0 * insideObjects.size() / outsideObjects.size() < 0.02) {
                index = (index + outsideObjects.size() / 7) % objectsToInsert.size();
                continue;
            }
            
            if (attempt != 4
                && insideObjects.size() > 10
                && 1.0 * outsideObjects.size() / insideObjects.size() < 0.02) {
                index = (index + outsideObjects.size() / 7) % objectsToInsert.size();
                continue;
            }

#if 0
            if (outsideObjects.empty() && insideObjects.size() > 10
                && index < objectsToInsert.size() - 1) {
                index += 1;
                continue;
            }
#endif

            std::sort(insideObjects.begin(), insideObjects.end());
            std::sort(outsideObjects.begin(), outsideObjects.end());
        
#if 0
            cerr << "depth = " << depth << " to insert " << objectsToInsert.size()
                 << " pivot items " << items.size()
                 << " inside " << insideObjects.size() << " outside "
                 << outsideObjects.size() << " firstNonZero " << firstNonZero
                 << " sp " << splitPoint << " radius " << radius << " first "
                 << sorted[firstNonZero].first << " last " << sorted.back().first
                 << endl;
            if (outsideObjects.empty() && sorted.size() < 200) {
                cerr << distances << endl;
            }
#endif

            std::unique_ptr<VantagePointTreeT> inside, outside;
            if (!insideObjects.empty())
                inside.reset(createParallel(insideObjects, distance, depth + 1));
            if (!outsideObjects.empty())
                outside.reset(createParallel(outsideObjects, distance, depth + 1));

            return new VantagePointTreeT(items, radius,
                                         std::move(inside), std::move(outside));
        }
    }

    /** Return the at most n closest neighbours, which must all have a
        distance of less than minimumRadius.
    */
    std::vector<std::pair<float, Item> >
    search(const std::function<float (Item)> & distance,
                      int n,
                      float maximumDist) const
    {
        using namespace std;

        std::vector<std::pair<float, Item> > result;

        // Add the results to the current set of nearest neighbours
        auto addResults = [&] (const std::vector<std::pair<float, Item> > & found)
            {
                // Insert into results list and look for the new maximum distance
                result.insert(result.end(), found.begin(), found.end());
                
                // Prune out solutions not viable
                std::sort(result.begin(), result.end());
                if (result.size() > n) {
                    result.resize(n);
                    maximumDist = result.back().first;
                }
            };

        // First, find the distance to the object at this node
        float pivotDistance = distance(items.at(0));
        
        //cerr << "search for " << pivotDistance << " maximumDist " << maximumDist
        //     << " inside " << !!inside << " outside " << !!outside
        //     << " clump " << clump.size() << endl;

        if (pivotDistance <= maximumDist) {
            for (auto & item: items)
                result.emplace_back(pivotDistance, item);
        }

        if (result.size() > n)
            result.resize(n);

        if (!clump.empty()) {
            ExcAssert(!inside);
            ExcAssert(!outside);

            // TODO: use radius to decide whether the clump is viable or not

            // Get the distance to each item in the clump
            std::vector<std::pair<float, Item> > clumpResult;
            
            for (auto & i: clump) {
                float dist = distance(i);
                if (dist <= maximumDist) {
                    clumpResult.emplace_back(dist, i);
                }
            }

            //cerr << "adding " << clumpResult.size() << " clump results" << " to "
            //     << result.size() << " existing" << endl;

            addResults(clumpResult);

            return result;
        }

        if (!inside && !outside)
            return result;

        const VantagePointTreeT * toSearchFirst;
        const VantagePointTreeT * toSearchSecond = nullptr;
        float closestPossibleSecond = INFINITY;

        // Choose which subtree to search first, and the condition for
        // searching the second one
        if (!inside)
            toSearchFirst = outside.get();
        else if (!outside)
            toSearchFirst = inside.get();
        else if (pivotDistance < radius) {
            toSearchFirst = inside.get();
            toSearchSecond = outside.get();
            closestPossibleSecond = radius - pivotDistance;
        }
        else {
            toSearchFirst = outside.get();
            toSearchSecond = inside.get();
            closestPossibleSecond = pivotDistance - radius;
        }

        addResults(toSearchFirst->search(distance, n, maximumDist));

        // We are conservative by this factor with distance comparisons, to
        // make the algorithm somewhat robust to slight numerical differences.
        float fudgeFactor = 1.00001f;

        if (toSearchSecond &&
            (result.size() < n || maximumDist * fudgeFactor >= closestPossibleSecond)) {
            addResults(toSearchSecond->search(distance, n, maximumDist));
        }

        return result;
    }

    /** Create a deep copy of the given node.  This also works for
        null pointers.
    */
    static VantagePointTreeT * deepCopy(const VantagePointTreeT * node)
    {
        if (!node)
            return nullptr;

        if (!node->clump.empty()) {
            ExcAssert(!node->inside);
            ExcAssert(!node->outside);
            return new VantagePointTreeT(node->items, node->radius, node->clump);
        }

        std::unique_ptr<VantagePointTreeT> inside(deepCopy(node->inside.get()));
        std::unique_ptr<VantagePointTreeT> outside(deepCopy(node->outside.get()));
        return new VantagePointTreeT(node->items, node->radius,
                                     std::move(inside), std::move(outside));
    }

    size_t memusage() const
    {
        return sizeof(*this)
            + sizeof(Item) * items.size()
            + sizeof(Item) * clump.size()
            + (inside ? inside->memusage() : 0)
            + (outside ? outside->memusage() : 0);
    }

    void serialize(DB::Store_Writer & store) const
    {
        store << items << radius << clump;
        serializePtr(store, inside.get());
        serializePtr(store, outside.get());
    }

    void reconstitute(DB::Store_Reader & store)
    {
        store >> items >> radius >> clump;
        inside.reset(reconstitutePtr(store));
        outside.reset(reconstitutePtr(store));
    }

    static void serializePtr(DB::Store_Writer & store, const VantagePointTreeT * ptr)
    {
        using namespace ML::DB;
        if (!ptr) {
            store << compact_size_t(0);
            return;
        }
        store << compact_size_t(1);
        ptr->serialize(store);
    }

    static VantagePointTreeT * reconstitutePtr(DB::Store_Reader & store)
    {
        using namespace ML::DB;
        compact_size_t present(store);
        if (!present)
            return nullptr;
        ExcAssertEqual(present, 1);
        std::unique_ptr<VantagePointTreeT> result(new VantagePointTreeT());
        result->reconstitute(store);
        return result.release();
    }
};

typedef VantagePointTreeT<int> VantagePointTree;

} // namespace ML
