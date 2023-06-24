"""
Autonomous localization of fiducial markers for IGNS.
This script runs the routine for the autonomous
detection and localization of fiducial markers affixed
on the skull, using a DICOM scan as input.
Package dependencies: dicom_numpy, pydicom, natsort,
scipy, sklearn, skimage and mayavi.
"""

from skullReconstruct import *
from skullNormalExtraction import *
from skullFindFiducial import *
import time
from mayavi import mlab
import copy

# ConstPixelSpacing = (1.0, 1.0, 1.0)

start_time = time.time()

PathDicom = "Fiducial Localization - MRI Scans-20220827T112322Z-001/Fiducial Localization - MRI Scans/2016.06.27 PVC Skull Model/Spiral Scan/DICOMDIR"

data = readDicomData(PathDicom)
voxelData, ConstPixelSpacing = get3DRecon(data, PathDicom)
print()
print("Constant Pixel Spacing: " + str(ConstPixelSpacing))
voxelData, ConstPixelSpacing = interpolate_image(
     voxelData, (3, 1, 1))  # interpolating the image
voxelDataThresh = applyThreshold(copy.deepcopy(voxelData))
print(ConstPixelSpacing)
print("---- %s seconds ----- Extracted %s Slices!" %
      (time.time() - start_time, str(voxelData.shape)))

surfaceVoxels = getSurfaceVoxels(voxelDataThresh)

print("---- %s seconds ----- Extracted Surface Voxels!" %
      (time.time() - start_time))

normals, surfaceVoxelCoord, verts, faces = findSurfaceNormals(copy.deepcopy(
    surfaceVoxels), voxelData, ConstPixelSpacing)

print("---- %s seconds ----- Extracted %s Surface Normals!" %
      (time.time() - start_time, len(surfaceVoxelCoord)))

sampling_factor = 20
normals_sample = normals[::sampling_factor]
surfaceVoxelCoord_sample = surfaceVoxelCoord[::sampling_factor]

surfaceVoxelCoord_sample = np.uint16(np.float64(
    surfaceVoxelCoord_sample) / ConstPixelSpacing)

print("---- %s seconds ----- Sampled %s Voxels!" %
      (time.time() - start_time, len(surfaceVoxelCoord_sample)))
costs, neighbourIndices = checkFiducial(surfaceVoxelCoord,
                               surfaceVoxelCoord_sample, normals, ConstPixelSpacing)

print("---- %s seconds ----- Finished comparing with Fiducial Model!" %
      (time.time() - start_time))

# Visualise in Mayavi
visualiseFiducials(costs, neighbourIndices, surfaceVoxelCoord_sample, surfaceVoxelCoord, verts, faces, num_markers=100)
