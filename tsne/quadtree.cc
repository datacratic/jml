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
              << centerOfMass;
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
              << child;
        return;
    }
    throw ML::Exception("Unknown quadtree node size");
}

void
QuadtreeNode::
reconstitute(DB::Store_Reader & store)
{
    throw ML::Exception("QuadtreeNode::reconstitute");
}

void
Quadtree::
serialize(DB::Store_Writer & store) const
{
    store << compact_size_t(0);  // version
    if (root) {
        store << compact_size_t(1);
        root->serialize(store);
    }
    else store << compact_size_t(0);
}

} // namespace ML

