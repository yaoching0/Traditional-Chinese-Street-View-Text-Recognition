from scipy.spatial import ConvexHull
from scipy.ndimage.interpolation import rotate
import pandas as pd
import numpy as np

#判斷center是否在df_bbx內
def is_in_bbx(df_bbx,center):#testPoint為待測點[x,y]
    testPoint=center.tolist()
    LBPoint = [df_bbx[0],df_bbx[1]]
    LTPoint = [df_bbx[2],df_bbx[1]]
    RTPoint = [df_bbx[2],df_bbx[3]]
    RBPoint = [df_bbx[0],df_bbx[3]]
    a = (LTPoint[0]-LBPoint[0])*(testPoint[1]-LBPoint[1])-(LTPoint[1]-LBPoint[1])*(testPoint[0]-LBPoint[0])
    b = (RTPoint[0]-LTPoint[0])*(testPoint[1]-LTPoint[1])-(RTPoint[1]-LTPoint[1])*(testPoint[0]-LTPoint[0])
    c = (RBPoint[0]-RTPoint[0])*(testPoint[1]-RTPoint[1])-(RBPoint[1]-RTPoint[1])*(testPoint[0]-RTPoint[0])
    d = (LBPoint[0]-RBPoint[0])*(testPoint[1]-RBPoint[1])-(LBPoint[1]-RBPoint[1])*(testPoint[0]-RBPoint[0])
    #print(a,b,c,d)
    if (a>0 and b>0 and c>0 and d>0) or (a<0 and b<0 and c<0 and d<0):
        return True
    else:
        return False

def computational_geometry_block(temp_assign_df):
    #回傳順時針四邊形
    #input pd.DataFrame
    #return [[x1,y1],[x2,y2],...]
    points_df=pd.concat([temp_assign_df[['xmin','ymin']].rename(columns={'xmin':'x', 'ymin':'y'}),
                    temp_assign_df[['xmax','ymin']].rename(columns={'xmax':'x', 'ymin':'y'}),
                    temp_assign_df[['xmax','ymax']].rename(columns={'xmax':'x', 'ymax':'y'}),
                    temp_assign_df[['xmin','ymax']].rename(columns={'xmin':'x', 'ymax':'y'})
                    ], axis=0).reset_index(drop=True)
    points=np.array(points_df)
    bbox = minimum_bounding_rectangle(points)
    bbox=bbox.tolist()
    return bbox[::-1]

def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    #source:https://stackoverflow.com/questions/13542855/algorithm-to-find-the-minimum-area-rectangle-for-given-points-in-order-to-comput/33619018#33619018
    
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
#     rotations = np.vstack([
#         np.cos(angles),
#         -np.sin(angles),
#         np.sin(angles),
#         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval

#計算兩個點的歐氏距離
def ed(m, n):
    return np.sqrt(np.sum((m - n) ** 2))

#判斷是否全是中文
def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True