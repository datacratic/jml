/** vantage_point_tree.h                                           -*- C++ -*-
    Jeremy Barnes, 18 November 2014
    Copyright (c) 2014 Datacratic Inc.  All rights reserved.

    Available under the BSD license, no attribution required.
*/

#pragma once

#include "jml/stats/distribution.h"
#include "jml/utils/exc_assert.h"
#include "jml/db/persistent.h"
#include <iostream>

namespace ML {

template<typename Item>
struct VantagePointTreeT {

    VantagePointTreeT(const std::vector<Item> & items, double radius,
                      std::unique_ptr<VantagePointTreeT> && inside,
                      std::unique_ptr<VantagePointTreeT> && outside)
        : items(items),
          radius(radius),
          inside(std::move(inside)), outside(std::move(outside))
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
        : items(1, item),
          radius(std::numeric_limits<float>::quiet_NaN())
    {
    }

    VantagePointTreeT(const std::vector<Item> & items)
        : items(items),
          radius(std::numeric_limits<float>::quiet_NaN())
    {
    }

    VantagePointTreeT()
        : radius(INFINITY)
    {
    }

    std::vector<Item> items;  // all these have zero distance from each other
    double radius;

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
    createParallel(const std::vector<Item> & objectsToInsert,
                   const std::function<ML::distribution<float> (Item, const std::vector<Item> &, int)> & distance,
                   int depth = 0)
    {
        using namespace std;

        if (objectsToInsert.empty())
            return nullptr;

        if (objectsToInsert.size() == 1)
            return new VantagePointTreeT(objectsToInsert[0]);

        // 1.  Choose a random object, in this case the first one
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
            sorted.emplace_back(distances[i], objectsToInsert[i]);
        }

        // Find the first one that's not zero
        std::vector<Item> items;
        size_t firstNonZero = 0;
        while (firstNonZero < sorted.size() && sorted[firstNonZero].first == 0.0) {
            items.push_back(sorted[firstNonZero].second);
            ++firstNonZero;
        }

        ExcAssertGreaterEqual(items.size(), 1);

        // If all have zero distance, just put them all in together
        if (firstNonZero == sorted.size()) {
            return new VantagePointTreeT(items);
        }

        // Get median distance, to use as a radius
        size_t splitPoint = firstNonZero + (distances.size() - firstNonZero) / 2;
        float radius = distances[splitPoint];
        
        // Split into two subgroups
        std::vector<Item> insideObjects;
        std::vector<Item> outsideObjects;

        for (unsigned i = firstNonZero;  i < objectsToInsert.size();  ++i) {
            if (sorted[i].first <= radius)
                insideObjects.push_back(sorted[i].second);
            else
                outsideObjects.push_back(sorted[i].second);
        }

        std::sort(insideObjects.begin(), insideObjects.end());
        std::sort(outsideObjects.begin(), outsideObjects.end());
        
        //cerr << "depth = " << depth << " to insert " << objectsToInsert.size()
        //     << " pivot items " << items.size()
        //     << " inside " << insideObjects.size() << " outside "
        //     << outsideObjects.size() << endl;

        std::unique_ptr<VantagePointTreeT> inside, outside;
        if (!insideObjects.empty())
            inside.reset(createParallel(insideObjects, distance, depth + 1));
        if (!outsideObjects.empty())
            outside.reset(createParallel(outsideObjects, distance, depth + 1));

        return new VantagePointTreeT(items, radius,
                                     std::move(inside), std::move(outside));
    }

    /** Return the at most n closest neighbours, which must all have a
        distance of less than minimumRadius.
    */
    std::vector<std::pair<float, Item> >
    search(const std::function<float (Item)> & distance,
                      int n,
                      float maximumDist) const
    {
        std::vector<std::pair<float, Item> > result;

        // First, find the distance to the object at this node
        float pivotDistance = distance(items.at(0));
        
        if (pivotDistance <= maximumDist) {
            for (auto & item: items)
                result.emplace_back(pivotDistance, item);
        }

        if (result.size() > n)
            result.resize(n);

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

    void serialize(DB::Store_Writer & store) const
    {
        store << items << radius;
        serializePtr(store, inside.get());
        serializePtr(store, outside.get());
    }

    void reconstitute(DB::Store_Reader & store)
    {
        store >> items >> radius;
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
