from typing import List
from collections import namedtuple
import time
import numpy as np


# Create 2D Point and print 2D Point objects
class Point(namedtuple("Point", "x y")):
    def __repr__(self) -> str:
        return f'Point{tuple(self)!r}'


class Rectangle(namedtuple("Rectangle", "lower upper")):
    # Create Rectangle and print Rectangle objects
    def __repr__(self) -> str:
        return f'Rectangle{tuple(self)!r}'
    
    # Determine if Point p is in the Rectangle
    def is_contains(self, p: Point) -> bool:
        return self.lower.x <= p.x <= self.upper.x and self.lower.y <= p.y <= self.upper.y
    

# Create Node class, print node
class Node(namedtuple("Node", "location left right")):
    """
    location: Point
    left: Node
    right: Node
    """
    def __repr__(self):
        return f'{tuple(self)!r}'

# Create KDTree class
class KDTree:
    """_n-d tree"""

    # Initialisation, storage of self._root
    def __init__(self):
        self._root = None
        self._n = 0

    # Insert Point list p
    def insert(self, p: List[Point]):
        """insert a list of points"""
        # Traversing the point list
        for i in p:
            # If tree is empty, the inserted point is set as self._root node
            if self._root is None:
                self._n += 1
                self._root = Node(i)
            # If tree is not empty, compare the value of x with self._root node
            else:
                # If x < x of self._root node, insert point into left subtree
                if self._root.location[self._n % 2] < i[self._n % 2]:
                    # If left subtree is empty, insert the point directly
                    if self._root.left is None:
                        self._n += 1
                        self._root.left = Node(i)
                    # If left subtree is not empty, insert by recursion,
                    # taking the self._root node of left subtree as the self._root node
                    # Since first dimension is x, next dimension is y, and repeats
                    else:
                        insert(self._root.left, i)
                # If x >= x of self._root node, insert point into right subtree
                else:
                    # If right subtree is empty, insert the point directly
                    if self._root.right is None:
                        self._n += 1
                        self._root.right = Node(i)
                    # If right subtree is not empty, insert by recursion,
                    # taking the self._root node of right subtree as the self._root node
                    else:
                        insert(self._root.right, i)

    # Search all points within specified rectangular interval
    def range(self, rectangle: Rectangle) -> List[Point]:
        """range query"""
        # Traversing the point list
        for j in List[Point]:
            # If the point is in the rectangle, return the original list
            if rectangle.is_contains(self, j) is True:
                return
            # If not, remove the point from the list and return new list
            else:
                List[Point].remove(j)
                return List[Point]

    def find_nearest(self, p:Point, axis=0, dist_func=lambda x, y: np.linalg.norm(x - y)):
        # If the root node is empty, then there is no nearest neighbour
        if self._root is None:
            self._best = None
 
        # If it is not leaf node, then continue to go down
        if self._root.left or self._root.right:
            new_axis = (axis + 1) % self._n
            if p[axis] < self._root.location[axis] and self._root.left:
                self.find_nearest(p, self._root.left, new_axis)
            elif self._root.right_child:
                self.find_nearest(p, self._root.right, new_axis)
 
        # Reverse and try to upload self._best
        dist = dist_func(self._root.location, p)
        if self._best is None or dist < self._best[0]:
            self._best = (dist, self._root.location)
 
        # If the diameter of query intersects other rectangles
        if abs(p[axis] - self._root.location[axis]) < self._best[0]:
            new_axis = (axis + 1) % self._n
            if self._root.left and p[axis] >= self._root.location[axis]:
                self.find_nearest(p, self._root.left, new_axis)
            elif self._root.right and p[axis] < self._root.location[axis]:
                self.find_nearest(p, self._root.right, new_axis)
 
        return self._best

        

# Range query test 
def range_test():
    points = [Point(7, 2), Point(5, 4), Point(9, 6), Point(4, 7), Point(8, 1), Point(2, 3)]
    kd = KDTree()
    kd.insert(points)
    result = kd.range(Rectangle(Point(0, 0), Point(6, 6)))
    assert sorted(result) == sorted([Point(2, 3), Point(5, 4)])

# Two kinds of query test
def performance_test():
    points = [Point(x, y) for x in range(1000) for y in range(1000)]

    lower = Point(500, 500)
    upper = Point(504, 504)
    rectangle = Rectangle(lower, upper)
    #  naive method
    start = int(round(time.time() * 1000))
    result1 = [p for p in points if rectangle.is_contains(p)]
    end = int(round(time.time() * 1000))
    print(f'Naive method: {end - start}ms')

    kd = KDTree()
    kd.insert(points)
    # _n-d tree
    start = int(round(time.time() * 1000))
    result2 = kd.range(rectangle)
    end = int(round(time.time() * 1000))
    print(f'K-D tree: {end - start}ms')

    assert sorted(result1) == sorted(result2)

# Run kdtree.py
if __name__ == '__main__':
    range_test()
    performance_test()