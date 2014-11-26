/** vantage_point_tree.h                                           -*- C++ -*-
    Jeremy Barnes, 18 November 2014
    Copyright (c) 2014 Datacratic Inc.  All rights reserved.

    Available under the BSD license, no attribution required.
*/

#pragma once

#include "jml/stats/distribution.h"

namespace ML {

struct VantagePointTree {

    VantagePointTree(int objectNumber, double radius,
                     std::unique_ptr<VantagePointTree> && inside,
                     std::unique_ptr<VantagePointTree> && outside)
        : objectNumber(objectNumber),
          radius(radius),
          inside(std::move(inside)), outside(std::move(outside))
    {
    }

    VantagePointTree(int objectNumber)
        : objectNumber(objectNumber),
          radius(std::numeric_limits<float>::quiet_NaN())
    {
    }

    int objectNumber;
    double radius;

    /// Children that are inside the ball of the given radius on object
    std::unique_ptr<VantagePointTree> inside;

    /// Children that are outside the ball of given radius on the object
    std::unique_ptr<VantagePointTree> outside;

    static VantagePointTree *
    create(const std::vector<int> & objectsToInsert,
           const std::function<float (int, int)> & distance)
    {
        if (objectsToInsert.empty())
            return nullptr;

        if (objectsToInsert.size() == 1)
            return new VantagePointTree(objectsToInsert[0]);

        // 1.  Choose a random object, in this case the first one
        int pivot = objectsToInsert[0];

        // Calculate distances to all children
        ML::distribution<float> distances(objectsToInsert.size());

        for (unsigned i = 1;  i < objectsToInsert.size();  ++i) {
            distances[i] = distance(pivot, objectsToInsert[i]);
        }
        
        ML::distribution<float> sorted(distances.begin(), distances.end());
        std::sort(sorted.begin() + 1, sorted.end());

        // Get median distance
        float radius = distances[distances.size() / 2];

        // Split into two subgroups
        std::vector<int> insideObjects;
        std::vector<int> outsideObjects;

        for (unsigned i = 1;  i < objectsToInsert.size();  ++i) {
            if (distances[i] < radius)
                insideObjects.push_back(objectsToInsert[i]);
            else
                outsideObjects.push_back(objectsToInsert[i]);
        }

        std::unique_ptr<VantagePointTree> inside, outside;
        if (!insideObjects.empty())
            inside.reset(create(insideObjects, distance));
        if (!outsideObjects.empty())
            outside.reset(create(outsideObjects, distance));

        return new VantagePointTree(pivot, radius,
                                    std::move(inside), std::move(outside));
    }

    /** Return the at most n closest neighbours, which must all have a
        distance of less than minimumRadius.
    */
    std::vector<std::pair<float, int> >
    search(const std::function<float (int)> & distance,
                      int n,
                      float maximumDist) const
    {
        std::vector<std::pair<float, int> > result;

        // First, find the distance to the object at this node
        float pivotDistance = distance(objectNumber);
        
        if (pivotDistance <= maximumDist)
            result.emplace_back(pivotDistance, objectNumber);

        if (!inside && !outside)
            return result;

        const VantagePointTree * toSearchFirst;
        const VantagePointTree * toSearchSecond = nullptr;
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

        // Add the results to the current set of nearest neighbours
        auto addResults = [&] (const std::vector<std::pair<float, int> > & found)
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

    size_t memusage() const
    {
        return sizeof(*this)
            + (inside ? inside->memusage() : 0)
            + (outside ? outside->memusage() : 0);
    }
};

} // namespace ML
