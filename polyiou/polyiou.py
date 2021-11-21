#!/usr/bin/python3
# -*- coding: UTF-8 -*-


from shapely.geometry import Polygon



p1 = Polygon([(0,0), (1,1), (1,0)])
p2 = Polygon([(0,1), (1,0), (1,1)])
print(p1.intersects(p2)) # judging if two polygons are intersected.
