/** quadtree.h                                                     -*- C++ -*-
    Jeremy Barnes, 17 November 2014
    Copyright (c) 2014 Datacratic Inc.  All rights reserved.

    Released under the BSD license, no attribution required.
*/

#pragma once

#include "jml/utils/compact_vector.h"
#include "jml/utils/exc_assert.h"

namespace ML {

typedef ML::compact_vector<float, 3> QCoord;

struct QuadtreeNode {

    /** Construct with a single child. */
    QuadtreeNode(QCoord mins, QCoord maxs, QCoord child)
        : mins(mins), maxs(maxs), type(TERMINAL),
          numChildren(1), centerOfMass(child), parentNodeNumber(-1),
          nodeNumber(-1)
    {
        center.resize(mins.size());
        for (unsigned i = 0;  i < mins.size();  ++i) {
            center[i] = 0.5 * (mins[i] + maxs[i]);
        }
    }

    /** Construct empty. */
    QuadtreeNode(QCoord mins, QCoord maxs)
        : mins(mins), maxs(maxs), type(EMPTY), numChildren(0),
          centerOfMass(mins.size()), parentNodeNumber(-1),
          nodeNumber(-1)
    {
        center.resize(mins.size());
        for (unsigned i = 0;  i < mins.size();  ++i) {
            center[i] = 0.5 * (mins[i] + maxs[i]);
        }
    }

    QCoord mins;   ///< Minimum coordinates for bounding box
    QCoord maxs;   ///< Maximum coordinates for bounding box
    QCoord center; ///< Cached pre-computation of center of bounding box

    enum Type {
        EMPTY = 0,     ///< Nothing in it
        NODE = 1,      ///< It's a node with child segments
        TERMINAL = 2   ///< It's a terminal node
    } type;

    int numChildren;   ///< Number of children in this part of tree
    QCoord centerOfMass;  ///< Center of mass of children, or child if terminal

    int parentNodeNumber;  ///< Node number of the parent node
    int nodeNumber;        ///< Number of this node; calculated by finish()

    /** The different quadrants for when we're a NODE. */
    std::map<int, std::unique_ptr<QuadtreeNode> > quadrants;

    /** Insert the given point into the tree. */
    void insert(QCoord point)
    {
        ExcAssertEqual(point.size(), mins.size());

        // Make sure that the point fits within the cell
        for (unsigned i = 0;  i < point.size();  ++i) {
            if (point[i] < mins[i] || point[i] > maxs[i]) {
                using namespace std;
                cerr << "point = " << point << endl;
                cerr << "mins " << mins << endl;
                cerr << "maxs " << maxs << endl;
                throw ML::Exception("point is not within cell");
            }
        }

        if (type == EMPTY) {
            // Easy case: first insertion into root of tree
            type = TERMINAL;
            centerOfMass = point;
            numChildren = 1;
        }
        else if (type == NODE) {
            // Insertion into an existing quad
            int quad = quadrant(center, point);
            auto it = quadrants.find(quad);
            if (it == quadrants.end()) {
                // Create a new quadrant
                QCoord newMins(point.size());
                QCoord newMaxs(point.size());

                for (unsigned i = 0;  i < point.size();  ++i) {
                    bool less = quad & (1 << i);

                    newMins[i] = less ? mins[i] : center[i];
                    newMaxs[i] = less ? center[i] : maxs[i];
                }

                quadrants[quad].reset(new QuadtreeNode(newMins, newMaxs, point));
            } else {
                // Recurse down into existing quadrant
                it->second->insert(point);
            }
            
            ++numChildren;
            
            for (unsigned i = 0;  i < point.size();  ++i) {
                centerOfMass[i] += point[i];
            }
        }
        else if (type == TERMINAL) {
            // First we convert to a non-terminal
            convertToNonTerminal();

            // Then we insert a new one
            insert(point);
        }
    }
    
    /** Walk the tree.  The function takes a QuadtreeNode and returns a
        bool as to whether to stop descending or not.
    */
    template<typename Fn>
    void walk(const Fn & fn, int depth = 0)
    {
        if (!fn(*this, depth))
            return;
        if (type == NODE) {
            for (auto & q: quadrants) {
                q.second->walk(fn, depth + 1);
            }
        }
    }

    /** Finish the structure, including calculating node numbers */
    int finish(int currentNodeNumber = 0, int parentNodeNumber = -1)
    {
        ExcAssertEqual(nodeNumber, -1);

        this->parentNodeNumber = parentNodeNumber;
        nodeNumber = currentNodeNumber;
        ++currentNodeNumber;
        
        for (auto & q: quadrants) {
            currentNodeNumber
                = q.second->finish(currentNodeNumber, this->nodeNumber);
        }
        
        return currentNodeNumber;
    }

    double diagonalLength() const
    {
        double result = 0.0;
        for (unsigned i = 0;  i < mins.size();  ++i) {
            float dist = maxs[i] - mins[i];
            result += dist * dist;
        }

        return sqrt(result);
    }

    /** Convert a node to a non-terminal. */
    void convertToNonTerminal()
    {
        ExcAssertEqual(type, TERMINAL);

        QCoord oldChild = centerOfMass;
        centerOfMass.clear();
        centerOfMass.resize(oldChild.size());
        numChildren = 0;

        // Convert to a non-terminal
        type = NODE;

        // Insert the current child and clear it
        insert(oldChild);
    }

    // Return which quadrant the given point is in
    static int quadrant(const QCoord & center, const QCoord & point)
    {
        ExcAssertEqual(center.size(), point.size());
        int result = 0;
        for (unsigned i = 0;  i < center.size();  ++i) {
            result = result | ((point[i] < center[i]) << i);
        }
        return result;
    }
};

struct Quadtree {

    Quadtree(QCoord mins, QCoord maxs)
    {
        root.reset(new QuadtreeNode(mins, maxs));
    }

    void insert(QCoord coord)
    {
        root->insert(coord);
    }

    std::unique_ptr<QuadtreeNode> root;
};

} // namespace ML
