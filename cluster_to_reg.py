#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division, absolute_import, print_function




import numpy as np
from scipy.spatial import Voronoi

import polygon as Polygon

def test():

    RadiusTot=1.
    Poly=np.array([[-RadiusTot,-RadiusTot],
                   [+RadiusTot,-RadiusTot],
                   [+RadiusTot,+RadiusTot],
                   [-RadiusTot,+RadiusTot]])*1

    Line=np.array([[0.,0.],
                   [5.,5.]])

    print((CutLineInside(Poly,Line)))

def GiveABLin(P0,P1):
    x0,y0=P0
    x1,y1=P1
    a=(y1-y0)/(x1-x0)
    b=y1-a*x1
    return b,a

def GiveB(P0,P1):
    x0,y0=P0
    x1,y1=P1

    B=np.array([x0-x1,y0-y1])
    B/=np.sqrt(np.sum(B**2))
    if B[0]<0: B=-B
    return B

def CutLineInside(Poly,Line):
    P=Polygon.Polygon(Poly)

    dx=1e-4
    PLine=np.array(Line.tolist()+Line.tolist()[::-1]).reshape((4,2))
    #PLine[2,0]+=dx
    #PLine[3,0]+=2*dx
    PLine[2:,:]+=np.random.randn(2,2)*1e-6
    P0=Polygon.Polygon(PLine)
    PP=np.array(P0&P)[0].tolist()
    PP.append(PP[0])

    B0=GiveB(Line[0],Line[1])
    B=[GiveB(PP[i],PP[i+1]) for i in range(len(PP)-1)]

    PLine=[]
    for iB in range(len(B)):
        d=np.sum((B[iB]-B0)**2)
        print((d,PP[iB],PP[iB+1]))
        if d==0:
            PLine.append([PP[iB],PP[iB+1]])


    LOut=np.array(PLine[0])

    return LOut


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue


        # reconstruct a non-finite region
        if p1 not in list(all_ridges.keys()):
            new_regions.append(vertices)
            continue
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n

            V=vor.vertices[v2]
            R=np.sqrt(np.sum(V**2))

            # while R>0.1:
            #     V*=0.9
            #     R=np.sqrt(np.sum(V**2))
            #     #print R

            vor.vertices[v2][:]=V[:]

            ThisRad=radius
            far_point = vor.vertices[v2] + direction * radius
            R=np.sqrt(np.sum(far_point**2))
            ThisRad=R

            # while R>1:
            #     ThisRad*=0.9
            #     far_point = vor.vertices[v2] + direction * ThisRad
            #     R=np.sqrt(np.sum(far_point**2))
            #     print "=============="
            #     print R,np.sqrt(np.sum(vor.vertices[v2]**2))
            #     print vor.vertices[v2]
            #     print direction
            #     print ThisRad
            #     #if R>1000:
            #     #    stop

            # RadiusTot=.3
            # Poly=np.array([[-RadiusTot,-RadiusTot],
            #                [+RadiusTot,-RadiusTot],
            #                [+RadiusTot,+RadiusTot],
            #                [-RadiusTot,+RadiusTot]])*1
            # stop
            # PT.CutLineInside(Poly,Line)


            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    regions, vertices=new_regions, np.asarray(new_vertices)

    return regions, vertices


HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
bold='\033[1m'
nobold='\033[0m'
Separator="================================%s=================================="
silent=0

def Str(strin0,col="red",Bold=True):
    if silent==1: return strin0
    strin=str(strin0)
    if col=="red":
        ss=FAIL
    if col=="green":
        ss=OKGREEN
    elif col=="yellow":
        ss=WARNING
    elif col=="blue":
        ss=OKBLUE
    elif col=="green":
        ss=OKGREEN
    elif col=="white":
        ss=""
    ss="%s%s%s"%(ss,strin,ENDC)
    if Bold: ss="%s%s%s"%(bold,ss,nobold)
    return ss

def Sep(strin=None,D=1):
    if D!=1:
        return Str(Separator%("="*len(strin)))
    else:
        return Str(Separator%(strin))

def Title(strin,Big=False):
    print()
    print()
    if Big: print((Sep(strin,D=0)))
    print((Sep(strin)))
    if Big: print((Sep(strin,D=0)))
    print()

def disable():
    HEADER = ''
    OKBLUE = ''
    OKGREEN = ''
    WARNING = ''
    FAIL = ''
    ENDC = ''


class ClassCoordConv():
    def __init__(self,rac,decc):

        rarad=rac
        decrad=decc
        self.rarad=rarad
        self.decrad=decrad
        cos=np.cos
        sin=np.sin
        mrot=np.array([[cos(rarad)*cos(decrad), sin(rarad)*cos(decrad),sin(decrad)],[-sin(rarad),cos(rarad),0.],[-cos(rarad)*sin(decrad),-sin(rarad)*sin(decrad),cos(decrad)]]).T
        vm=np.array([[0.,0.,1.]]).T
        vl=np.array([[0.,1., 0.]]).T
        vn=np.array([[1., 0, 0.]]).T
        self.vl2=np.dot(mrot,vl)
        self.vm2=np.dot(mrot,vm)
        self.vn2=np.dot(mrot,vn)
        self.R=np.array([[cos(decrad)*cos(rarad),cos(decrad)*sin(rarad),sin(decrad)]]).T


    def lm2radec(self,l_list,m_list):

        ra_list=np.zeros(l_list.shape,dtype=np.float)
        dec_list=np.zeros(l_list.shape,dtype=np.float)

        for i in range(l_list.shape[0]):
            l=l_list[i]
            m=m_list[i]
            if (l_list[i]==0.)&(m_list[i]==0.):
                ra_list[i]=self.rarad
                dec_list[i]=self.decrad
                continue
            Rp=self.R+self.vl2*l+self.vm2*m-(1.-np.sqrt(1.-l**2-m**2))*self.vn2
            dec_list[i]=np.arcsin(Rp[2])
            ra_list[i]=np.arctan(Rp[1]/Rp[0])
            if Rp[0]<0.: ra_list[i]+=np.pi

        return ra_list,dec_list

    def radec2lm(self,ra,dec):
        l = np.cos(dec) * np.sin(ra - self.rarad)
        m = np.sin(dec) * np.cos(self.decrad) - np.cos(dec) * np.sin(self.decrad) * np.cos(ra - self.rarad)
        return l,m





class VoronoiToReg():
    def __init__(self,rac,decc):
        self.rac=rac
        self.decc=decc
        self.CoordMachine=ClassCoordConv(rac,decc)

    def ToReg(self,regFile,xc,yc,radius=0.1,Col="red"):
        print("Writing voronoi in: %s"%Str(regFile,col="blue"))
        f=open(regFile,"w")
        f.write("# Region file format: DS9 version 4.1\n")
        ss0='global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0'
        ss1=' fixed=0 edit=1 move=1 delete=1 include=1 source=1\n'

        f.write(ss0+ss1)
        f.write("fk5\n")

        CoordMachine=self.CoordMachine

        xy=np.zeros((xc.size,2),np.float32)
        xy[:,0]=xc
        xy[:,1]=yc
        vor = Voronoi(xy)
        regions, vertices = voronoi_finite_polygons_2d(vor,radius=radius)


        for region in regions:
            polygon0 = vertices[region]
            P=polygon0.tolist()
            polygon=np.array(P+[P[0]])
            for iline in range(polygon.shape[0]-1):

                x0,y0=CoordMachine.lm2radec(np.array([polygon[iline][0]]),np.array([polygon[iline][1]]))
                x1,y1=CoordMachine.lm2radec(np.array([polygon[iline+1][0]]),np.array([polygon[iline+1][1]]))

                x0*=180./np.pi
                y0*=180./np.pi
                x1*=180./np.pi
                y1*=180./np.pi


                f.write("line(%f,%f,%f,%f) # line=0 0 color=%s dash=1\n"%(x0,y0,x1,y1,Col))
                #f.write("line(%f,%f,%f,%f) # line=0 0 color=red dash=1\n"%(x1,y0,x0,y1))

        f.close()


    def VorToReg(self,regFile,vor,radius=0.1,Col="red"):
        print("Writing voronoi in: %s"%Str(regFile,col="blue"))

        f=open(regFile,"w")
        f.write("# Region file format: DS9 version 4.1\n")
        ss0='global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0'
        ss1=' fixed=0 edit=1 move=1 delete=1 include=1 source=1\n'

        f.write(ss0+ss1)
        f.write("fk5\n")

        CoordMachine=self.CoordMachine

        regions, vertices = vor.regions,vor.vertices


        for region in regions:
            if len(region)==0: continue
            polygon0 = vertices[region]
            P=polygon0.tolist()
            polygon=np.array(P+[P[0]])
            for iline in range(polygon.shape[0]-1):

                x0,y0=CoordMachine.lm2radec(np.array([polygon[iline][0]]),np.array([polygon[iline][1]]))
                x1,y1=CoordMachine.lm2radec(np.array([polygon[iline+1][0]]),np.array([polygon[iline+1][1]]))

                x0*=180./np.pi
                y0*=180./np.pi
                x1*=180./np.pi
                y1*=180./np.pi

                f.write("line(%f,%f,%f,%f) # line=0 0 color=%s dash=1\n"%(x0,y0,x1,y1,Col))
                #f.write("line(%f,%f,%f,%f) # line=0 0 color=red dash=1\n"%(x1,y0,x0,y1))

        f.close()

    def PolygonToReg(self,regFile,LPolygon,radius=0.1,Col="red",labels=None):
        print("Writing voronoi in: %s"%Str(regFile,col="blue"))

        f=open(regFile,"w")
        f.write("# Region file format: DS9 version 4.1\n")
        ss0='global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0'
        ss1=' fixed=0 edit=1 move=1 delete=1 include=1 source=1\n'

        f.write(ss0+ss1)
        f.write("fk5\n")

        CoordMachine=self.CoordMachine


        for iFacet,polygon0 in zip(list(range(len(LPolygon))),LPolygon):
            #polygon0 = vertices[region]
            P=polygon0.tolist()
            if len(polygon0)==0: continue
            polygon=np.array(P+[P[0]])
            ThisText=""
            if labels is not None:
                lmean0=np.mean(polygon[:,0])
                mmean0=np.mean(polygon[:,1])

                lmean,mmean,ThisText=labels[iFacet]
                # print "!!!!======"
                # print lmean0,mmean0
                # print lmean,mmean

                xm,ym=CoordMachine.lm2radec(np.array([lmean]),np.array([mmean]))
                xm*=180./np.pi
                ym*=180./np.pi
                f.write("point(%f,%f) # text={%s} point=circle 5 color=red width=2\n"%(xm,ym,ThisText))

            for iline in range(polygon.shape[0]-1):


                L0,M0=np.array([polygon[iline][0]]),np.array([polygon[iline][1]])
                x0,y0=CoordMachine.lm2radec(L0,M0)
                L1,M1=np.array([polygon[iline+1][0]]),np.array([polygon[iline+1][1]])
                x1,y1=CoordMachine.lm2radec(L1,M1)

                x0*=180./np.pi
                y0*=180./np.pi
                x1*=180./np.pi
                y1*=180./np.pi

                # print "===================="
                # print "[%3.3i] %f %f %f %f"%(iline,x0,y0,x1,y1)
                # print "       %s"%str(L0)
                # print "       %s"%str(L1)
                # print "       %s"%str(M0)
                # print "       %s"%str(M1)
                f.write("line(%f,%f,%f,%f) # line=0 0 color=%s dash=1 \n"%(x0,y0,x1,y1,Col))

                #f.write("line(%f,%f,%f,%f) # line=0 0 color=red dash=1\n"%(x1,y0,x0,y1))

        f.close()