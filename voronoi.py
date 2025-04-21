#############################################################################
#
# Voronoi diagram calculator with progress tracking
# Based on Python code by Bill Simons (2005) and Carson Farmer (2010)
# Original derived from Steven Fortune's algorithm
#
#############################################################################

import math
import sys
import getopt
from tqdm import tqdm
import numpy as np

TOLERANCE = 1e-9
BIG_FLOAT = 1e38

#----------------------------------------------------------------------------
class Context(object):
    def __init__(self):
        self.doPrint = 0
        self.plot = 0
        self.triangulate = False
        self.vertices = [] # list of vertex 2-tuples: (x,y)
        self.lines = [] # equation of line 3-tuple (a b c), for the equation of the line a*x+b*y = c  
        self.edges = [] # edge 3-tuple: (line index, vertex 1 index, vertex 2 index)   if either vertex index is -1, the edge extends to infinity
        self.triangles = [] # 3-tuple of vertex indices
        self.polygons = {} # a dict of site:[edges] pairs

    def out_site(self, s):
        self.vertices.append((s.x, s.y))

    def out_vertex(self, s):
        self.vertices.append((s.x, s.y))

    def out_triple(self, s1, s2, s3):
        self.triangles.append((s1, s2, s3))

    def out_bisector(self, edge):
        self.lines.append(edge.a, edge.b, edge.c)

    def out_edge(self, edge):

        site_numL = -1
        if edge.ep[Edge.LE] is not None:
            site_numL = edge.ep[Edge.LE].siteNum

        site_numR = -1
        if edge.ep[Edge.RE] is not None:
            site_numR = edge.ep[Edge.RE].siteNum

        if edge.reg[0].sitnum not in self.polygons:
            self.polygons[edge.reg[0].sitnum] = []

        if edge.reg[1].sitnum not in self.polygons:
            self.polygons[edge.reg[1].sitnum] = []
        
        self.polygons[edge.reg[0].sitnum].append((edge, site_numL, site_numR))
        self.polygons[edge.reg[1].sitnum].append((edge, site_numR, site_numL))
        self.edges.append((edge.edgenum, site_numL, site_numR))
        
#----------------------------------------------------------------------------
def voronoi(site_list, context):
    """
    Compute the Voronoi diagram with progress tracking.
    
    Args:
        siteList: List of site points
        context: Context object to store results
    """
    try:
        edge_list = EdgeList(site_list.xmin, site_list.xmax, len(site_list))
        priority_Q = PriorityQueue(site_list.ymin, site_list.ymax, len(site_list))

        # Initialize site iterator
        site_iter = site_list.iterator()

        # Get first site (bottom-most)
        bottomsite = site_iter.next()
        context.out_site(bottomsite)

        # Initialize with second site
        newsite = site_iter.next()

        # Placeholder for min point
        minpt = Site(-BIG_FLOAT, -BIG_FLOAT)

        # Count of total sites for progress
        total_sites = len(site_list)

        # Initialise progress bar for site processing
        with tqdm(total=total_sites, desc="Processing sites") as site_pbar:
            site_pbar.update(2) # Already processed 2 sites (bottomsite and newsite)

            # Main loop - process all sites and handle events
            while True:
                if not priority_Q.isEmpty():
                    minpt = priority_Q.getMinPt()

                # Process site events first (when new site is smaller than minpt)
                if (newsite and (priority_Q.isEmpty() or cmp(newsite, minpt) < 0)):
                    # This is a site event - process the new site
                    context.outSite(newsite)

                    # Get first Halfedge to the LEFT and RIGHT of the new site 
                    lbnd = edge_list.leftbnd(newsite) 
                    rbnd = lbnd.right   

                    # Create a new edge
                    bot = lbnd.rightreg(bottomsite)     
                    edge = Edge.bisect(bot, newsite)      
                    context.outBisector(edge)

                    # Create and insert new Halfedge
                    bisector = Halfedge(edge, Edge.LE)
                    edge_list.insert(lbnd, bisector)

                    # Check for intersection
                    p = lbnd.intersect(bisector)
                    if p is not None:
                        priority_Q.delete(lbnd)
                        priority_Q.insert(lbnd, p, newsite.distance(p))

                    # Create and insert another Halfedge
                    lbnd = bisector
                    bisector = Halfedge(edge, Edge.RE)     
                    edge_list.insert(lbnd, bisector)

                    # Check for intersection
                    p = bisector.intersect(rbnd)
                    if p is not None:
                        priority_Q.insert(bisector, p, newsite.distance(p))
                    
                    # Get next site
                    newsite = site_iter.next()
                    # Update progress bar
                    site_pbar.update(1)

                elif not priority_Q.isEmpty():
                    # This is a vector (circle) event
                    # Pop the Halfedge with the lowest vector
                    lbnd = priority_Q.popMinHalfedge()
                    llbnd = lbnd.left
                    rbnd = lbnd.right
                    rrbnd = rbnd.right

                    # Get relevant sites
                    bot = lbnd.leftreg(bottomsite)
                    top = lbnd.rightreg(bottomsite)

                    # Output triple
                    mid = lbnd.rightreg(bottomsite)
                    context.outTriple(bot, mid, top)

                    # Process vertex
                    v = lbnd.vertex
                    site_list.setSiteNumber(v)
                    context.outVertex(v)

                    # Set endpoints
                    if lbnd.edge.setEndpoint(lbnd.pm, v):
                        context.outEdge(lbnd.edge)
                    
                    if rbnd.edge.setEndpoint(rbnd.pm, v):
                        context.outEdge(rbnd.edge)

                    # Clean up halfedges
                    edge_list.delete(lbnd)           
                    priority_Q.delete(rbnd)
                    edge_list.delete(rbnd)

                    # determine orientation
                    pm = Edge.LE
                    if bot.y > top.y:
                        bot, top = top, bot
                        pm = Edge.RE

                    # Create new edge
                    edge = Edge.bisect(bot, top)     
                    context.outBisector(edge)

                    # Create halfedge from the edge 
                    bisector = Halfedge(edge, pm)    
                    
                    # Insert and set endpoint
                    edge_list.insert(llbnd, bisector) 
                    if edge.setEndpoint(Edge.RE - pm, v):
                        context.outEdge(edge)
                    
                    # Check for new intersections
                    p = llbnd.intersect(bisector)
                    if p is not None:
                        priority_Q.delete(llbnd);
                        priority_Q.insert(llbnd, p, bot.distance(p))

                    p = bisector.intersect(rrbnd)
                    if p is not None:
                        priority_Q.insert(bisector, p, bot.distance(p))
                else:
                    # We're done with all events
                    break

            print("Finalizing edges...")
            he = edge_list.leftend.right
            with tqdm(desc="Finalizing edges") as edge_pbar:
                while he is not edge_list.rightend:
                    context.outEdge(he.edge)
                    he = he.right
                    edge_pbar.update(1)
                    
            Edge.EDGE_NUM = 0
            print("Voronoi diagram generation complete!")

    except Exception as err:
        print("Error in Voronoi algorithm:")
        print(str(err))
        import traceback
        traceback.print_exc()

#----------------------------------------------------------------------------
def is_equal(a, b, relativeError=TOLERANCE):
    # is nearly equal to within the allowed relative error
    norm = max(abs(a), abs(b))
    return (norm < relativeError) or (abs(a - b) < (relativeError * norm))

# Comparison function for Python 3 compatibility due to the adaptation from old Python 2 code
def cmp(a, b):
    return (a > b) - (a < b)

#----------------------------------------------------------------------------
class Site(object):
    def __init__(self, x=0.0, y=0.0, site_num=0):
        self.x = x
        self.y = y
        self.site_num = site_num

    def __lt__(self, other):
        if self.y < other.y:
            return True
        elif self.y > other.y:
            return False
        elif self.x < other.x:
            return True
        elif self.x > other.x:
            return False
        else:
            return False
    
    def distance(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx*dx + dy*dy)
    
#----------------------------------------------------------------------------
class Edge(object):
    LE = 0
    RE = 1
    EDGE_NUM = 0
    DELETED = {}   # marker value

    def __init__(self):
        self.a = 0.0
        self.b = 0.0
        self.c = 0.0
        self.ep = [None, None]
        self.reg = [None, None]
        self.edgenum = 0

    def set_endpoint(self, lrFlag, site):
        self.ep[lrFlag] = site
        if self.ep[Edge.RE - lrFlag] is None:
            return False
        return True
    
    @staticmethod
    def bisect(s1, s2):
        newedge = Edge()
        newedge.reg[0] = s1 # store the sites that this edge is bisecting
        newedge.reg[1] = s2

        # to begin with, there are no endpoints on the bisector - it goes to infinity
        # ep[0] and ep[1] are None

        # get the difference in x dist between the sites
        dx = float(s2.x - s1.x)
        dy = float(s2.y - s1.y)
        adx = abs(dx)  # make sure that the difference in positive
        ady = abs(dy)
        
        # get the slope of the line
        newedge.c = float(s1.x * dx + s1.y * dy + (dx*dx + dy*dy)*0.5)  
        if adx > ady:
            # set formula of line, with x fixed to 1
            newedge.a = 1.0
            newedge.b = dy/dx
            newedge.c /= dx
        else:
            # set formula of line, with y fixed to 1
            newedge.b = 1.0
            newedge.a = dx/dy
            newedge.c /= dy

        newedge.edgenum = Edge.EDGE_NUM
        Edge.EDGE_NUM += 1
        return newedge
    
#------------------------------------------------------------------
class Halfedge(object):
    def __init__(self, edge=None, pm=Edge.LE):
        self.left = None   # left Halfedge in the edge list
        self.right = None  # right Halfedge in the edge list
        self.qnext = None  # priority queue linked list pointer
        self.edge = edge   # edge list Edge
        self.pm = pm
        self.vertex = None  # Site()
        self.ystar = BIG_FLOAT

    def __cmp__(self, other):
        if self.ystar > other.ystar:
            return 1
        elif self.ystar < other.ystar:
            return -1
        elif self.vertex.x > other.vertex.x:
            return 1
        elif self.vertex.x < other.vertex.x:
            return -1
        else:
            return 0

    def leftreg(self, default):
        if not self.edge: 
            return default
        elif self.pm == Edge.LE:
            return self.edge.reg[Edge.LE]
        else:
            return self.edge.reg[Edge.RE]

    def rightreg(self, default):
        if not self.edge: 
            return default
        elif self.pm == Edge.LE:
            return self.edge.reg[Edge.RE]
        else:
            return self.edge.reg[Edge.LE]

    # returns True if p is to right of halfedge self
    def isPointRightOf(self, pt):
        e = self.edge
        topsite = e.reg[1]
        right_of_site = pt.x > topsite.x
        
        if(right_of_site and self.pm == Edge.LE): 
            return True
        
        if(not right_of_site and self.pm == Edge.RE):
            return False
        
        if(e.a == 1.0):
            dyp = pt.y - topsite.y
            dxp = pt.x - topsite.x
            fast = 0;
            if ((not right_of_site and e.b < 0.0) or (right_of_site and e.b >= 0.0)):
                above = dyp >= e.b * dxp
                fast = above
            else:
                above = pt.x + pt.y * e.b > e.c
                if(e.b < 0.0):
                    above = not above
                if (not above):
                    fast = 1
            if (not fast):
                dxs = topsite.x - (e.reg[0]).x
                above = e.b * (dxp*dxp - dyp*dyp) < dxs*dyp*(1.0+2.0*dxp/dxs + e.b*e.b)
                if(e.b < 0.0):
                    above = not above
        else:  # e.b == 1.0 
            yl = e.c - e.a * pt.x
            t1 = pt.y - yl
            t2 = pt.x - topsite.x
            t3 = yl - topsite.y
            above = t1*t1 > t2*t2 + t3*t3
        
        if(self.pm == Edge.LE):
            return above
        else:
            return not above
        
    # create a new site where the Halfedges el1 and el2 intersect
    def intersect(self, other):
        e1 = self.edge
        e2 = other.edge
        if (e1 is None) or (e2 is None):
            return None

        # if the two edges bisect the same parent return None
        if e1.reg[1] is e2.reg[1]:
            return None

        d = e1.a * e2.b - e1.b * e2.a
        if is_equal(d, 0.0):
            return None

        xint = (e1.c*e2.b - e2.c*e1.b) / d
        yint = (e2.c*e1.a - e1.c*e2.a) / d
        if(cmp(e1.reg[1], e2.reg[1]) < 0):
            he = self
            e = e1
        else:
            he = other
            e = e2

        rightOfSite = xint >= e.reg[1].x
        if((rightOfSite and he.pm == Edge.LE) or
           (not rightOfSite and he.pm == Edge.RE)):
            return None

        # create a new site at the point of intersection - this is a new 
        # vector event waiting to happen
        return Site(xint, yint)
    
#------------------------------------------------------------------
class EdgeList(object):
    def __init__(self, xmin, xmax, nsites):
        if xmin > xmax: xmin, xmax = xmax, xmin
        self.hashsize = int(2*math.sqrt(nsites+4))
        
        self.xmin = xmin
        self.deltax = float(xmax - xmin)
        self.hash = [None]*self.hashsize
        
        self.leftend = Halfedge()
        self.rightend = Halfedge()
        self.leftend.right = self.rightend
        self.rightend.left = self.leftend
        self.hash[0] = self.leftend
        self.hash[-1] = self.rightend

    def insert(self, left, he):
        he.left = left
        he.right = left.right
        left.right.left = he
        left.right = he

    def delete(self, he):
        he.left.right = he.right
        he.right.left = he.left
        he.edge = Edge.DELETED

    # Get entry from hash table, pruning any deleted nodes 
    def gethash(self, b):
        if(b < 0 or b >= self.hashsize):
            return None
        he = self.hash[b]
        if he is None or he.edge is not Edge.DELETED:
            return he

        #  Hash table points to deleted half edge.  Patch as necessary.
        self.hash[b] = None
        return None

    def leftbnd(self, pt):
        # Use hash table to get close to desired halfedge 
        bucket = int(((pt.x - self.xmin)/self.deltax * self.hashsize))
        
        if(bucket < 0): 
            bucket = 0;
        
        if(bucket >= self.hashsize): 
            bucket = self.hashsize-1

        he = self.gethash(bucket)
        if(he is None):
            i = 1
            while True:
                he = self.gethash(bucket-i)
                if (he is not None): break;
                he = self.gethash(bucket+i)
                if (he is not None): break;
                i += 1
    
        # Now search linear list of halfedges for the correct one
        if (he is self.leftend) or (he is not self.rightend and he.isPointRightOf(pt)):
            he = he.right
            while he is not self.rightend and he.isPointRightOf(pt):
                he = he.right
            he = he.left;
        else:
            he = he.left
            while (he is not self.leftend and not he.isPointRightOf(pt)):
                he = he.left

        # Update hash table and reference counts
        if(bucket > 0 and bucket < self.hashsize-1):
            self.hash[bucket] = he
        return he

#------------------------------------------------------------------
class PriorityQueue(object):
    def __init__(self, ymin, ymax, nsites):
        self.ymin = ymin
        self.deltay = ymax - ymin
        self.hashsize = int(4 * math.sqrt(nsites))
        self.count = 0
        self.minidx = 0
        self.hash = []
        for i in range(self.hashsize):
            self.hash.append(Halfedge())

    def __len__(self):
        return self.count

    def isEmpty(self):
        return self.count == 0

    def insert(self, he, site, offset):
        he.vertex = site
        he.ystar = site.y + offset
        last = self.hash[self.getBucket(he)]
        next = last.qnext
        while((next is not None) and cmp(he, next) > 0):
            last = next
            next = last.qnext
        he.qnext = last.qnext
        last.qnext = he
        self.count += 1

    def delete(self, he):
        if (he.vertex is not None):
            last = self.hash[self.getBucket(he)]
            while last.qnext is not he:
                last = last.qnext
            last.qnext = he.qnext
            self.count -= 1
            he.vertex = None

    def getBucket(self, he):
        bucket = int(((he.ystar - self.ymin) / self.deltay) * self.hashsize)
        if bucket < 0: bucket = 0
        if bucket >= self.hashsize: bucket = self.hashsize-1
        if bucket < self.minidx: self.minidx = bucket
        return bucket

    def getMinPt(self):
        while(self.hash[self.minidx].qnext is None):
            self.minidx += 1
        he = self.hash[self.minidx].qnext
        x = he.vertex.x
        y = he.ystar
        return Site(x, y)

    def popMinHalfedge(self):
        curr = self.hash[self.minidx].qnext
        self.hash[self.minidx].qnext = curr.qnext
        self.count -= 1
        return curr

#------------------------------------------------------------------
class SiteList(object):
    def __init__(self, pointList):
        self.__sites = []
        self.__sitenum = 0

        self.__xmin = pointList[0].x
        self.__ymin = pointList[0].y
        self.__xmax = pointList[0].x
        self.__ymax = pointList[0].y
        for i, pt in enumerate(pointList):
            self.__sites.append(Site(pt.x, pt.y, i))
            if pt.x < self.__xmin: self.__xmin = pt.x
            if pt.y < self.__ymin: self.__ymin = pt.y
            if pt.x > self.__xmax: self.__xmax = pt.x
            if pt.y > self.__ymax: self.__ymax = pt.y
        self.__sites.sort()

    def setSiteNumber(self, site):
        site.sitenum = self.__sitenum
        self.__sitenum += 1

    class Iterator(object):
        def __init__(this, lst):  this.generator = (s for s in lst)
        def __iter__(this):      return this
        def next(this): 
            try:
                return next(this.generator)  # Use the next() function instead
            except StopIteration:
                return None
        # Python 3 compatibility
        def __next__(this):
            return this.next()

    def iterator(self):
        return SiteList.Iterator(self.__sites)

    def __iter__(self):
        return SiteList.Iterator(self.__sites)

    def __len__(self):
        return len(self.__sites)

    def _getxmin(self): return self.__xmin
    def _getymin(self): return self.__ymin
    def _getxmax(self): return self.__xmax
    def _getymax(self): return self.__ymax
    xmin = property(_getxmin)
    ymin = property(_getymin)
    xmax = property(_getxmax)
    ymax = property(_getymax)

#------------------------------------------------------------------
def computeVoronoiDiagram(points):
    """ 
    Takes a list of point objects (which must have x and y fields).
    Returns a 3-tuple of:

       (1) a list of 2-tuples, which are the x,y coordinates of the 
           Voronoi diagram vertices
       (2) a list of 3-tuples (a,b,c) which are the equations of the
           lines in the Voronoi diagram: a*x + b*y = c
       (3) a list of 3-tuples, (l, v1, v2) representing edges of the 
           Voronoi diagram.  l is the index of the line, v1 and v2 are
           the indices of the vertices at the end of the edge.  If 
           v1 or v2 is -1, the line extends to infinity.
    """
    print(f"Computing Voronoi diagram for {len(points)} points...")
    siteList = SiteList(points)
    context = Context()
    voronoi(siteList, context)
    return (context.vertices, context.lines, context.edges)