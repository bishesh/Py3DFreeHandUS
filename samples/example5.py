

from Py3DFreeHandUS.kine import *

# Read C3D file
fileName = 'ncc2_dave1.c3d'
markers = readC3D(fileName, ['markers'])['markers']

# Calculate reference frame without SVD
mkrList = ('Rigid_Body_1-Marker_1','Rigid_Body_1-Marker_2','Rigid_Body_1-Marker_3','Rigid_Body_1-Marker_4')
R, T = markersClusterFun(markers, mkrList)

# Invert roto-translation matrix
Rfull = composeRotoTranslMatrix(R, T)
Rfull = np.linalg.inv(Rfull)

# Express markers in the local rigid probe reference fram
markersLoc = changeMarkersReferenceFrame(markers, Rfull)

# Calculate reference frame with SVD
idxLoc = 1000   # Time frame from which to create the rigid body template
args = {}
markersLoc = {m: markersLoc[m][idxLoc,:] for m in mkrList}
print 'Position of markers in rigid reference frame for time frame %d:\n%s' % (idxLoc, markersLoc)
args['mkrsLoc'] = markersLoc
Rsvd, Tsvd = rigidBodySVDFun(markers, mkrList, args)

# Show roto-translation matrices (no SVD vs SVD) for one time frame
idx = 500   # time frame
print 'Rotation (no SVD) at frame %d:\n%s' % (idx, R[idx])
print 'Translation (no SVD) at frame %d:\n%s' % (idx, T[idx])
print 'Rotation (SVD) at frame %d:\n%s' % (idx, Rsvd[idx])
print 'Translation (SVD) at frame %d:\n%s' % (idx, Tsvd[idx])