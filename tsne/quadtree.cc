/** quadtree.cc
    Jeremy Barnes, 6 February 2015
    Copyright (c) 2015 Datacratic Inc.
    
    Released under the BSD license, no attribution required.
*/

#include "quadtree.h"
#include "jml/db/persistent.h"

using namespace ML::DB;

namespace ML {

void
QuadtreeNode::
serialize(DB::Store_Writer & store) const
{
    switch (type) {
    case EMPTY:
        store << compact_size_t(0);
        return;
    case NODE:
        store << compact_size_t(1);
        store << compact_size_t(numChildren)
              << centerOfMass << mins << maxs;
        for (auto & q: quadrants) {
            if (!q) {
                store << compact_size_t(0);
            }
            else {
                store << compact_size_t(1);
                q->serialize(store);
            }
        }
        return;
    case TERMINAL:
        store << compact_size_t(2);
        store << compact_size_t(numChildren)
              << child << mins << maxs;
        return;
    }
    throw ML::Exception("Unknown quadtree node size");
}

QuadtreeNode::
QuadtreeNode(DB::Store_Reader & store, int version)
    : diag(0.0), type(EMPTY), numChildren(0), recipNumChildren{0, 0}
{
    if (version != 0)
        throw ML::Exception("Unknown quadtree node version");

    DB::compact_size_t type(store);
    if (type == 0) {
        throw ML::Exception("Reconstituting an empty quadtree node");
    }
    else if (type == 1) {
        type = NODE;
        compact_size_t nc(store);
        numChildren = nc;
        store >> centerOfMass >> mins >> maxs;
        child.clear();

        quadrants.resize(1 << centerOfMass.size());

        for (auto & q: quadrants) {
            compact_size_t indicator(store);
            if (indicator == 0) {
                q = nullptr;
                continue;
            }
            ExcAssertEqual(indicator, 1);
            q = new QuadtreeNode(store, version);
        }
    }
    else if (type == 2) {
        type = TERMINAL;
        compact_size_t nc(store);
        numChildren = nc;
        store >> child >> mins >> maxs;
        mins = maxs = center = centerOfMass = child;

    }
    else throw ML::Exception("Unknown quadtree node type");

    center.resize(mins.size());
    for (unsigned i = 0;  i < mins.size();  ++i) {
        center[i] = 0.5 * (mins[i] + maxs[i]);
        ExcAssert(std::isfinite(mins[i]));
        ExcAssert(std::isfinite(maxs[i]));
        ExcAssert(std::isfinite(child[i]));
    }

    diag = diagonalLength();
    recipNumChildren[0] = 1.0 / numChildren;
    recipNumChildren[1] = 1.0 / (numChildren - 1);
}

Quadtree::
Quadtree(DB::Store_Reader & store)
{
    std::string canary;
    store >> canary;
    if (canary != "QTREE")
        throw ML::Exception("Unknown quadtree canary");
    DB::compact_size_t version(store);
    if (version != 0)
        throw ML::Exception("Unknown quadtree version");

    DB::compact_size_t indicator(store);
    if (indicator == 0)
        return;
    else if (indicator == 1) {
        root.reset(new QuadtreeNode(store, version));
    }
    else throw ML::Exception("Unknown quadtree indicator");
}

void
Quadtree::
serialize(DB::Store_Writer & store) const
{
    store << std::string("QTREE") << compact_size_t(0);  // version
    if (root) {
        store << compact_size_t(1);
        root->serialize(store);
    }
    else store << compact_size_t(0);
}

} // namespace ML

