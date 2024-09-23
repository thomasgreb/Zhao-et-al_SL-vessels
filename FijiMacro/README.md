# Zhao et al. Fiji macro

## System requirements
- The macro requires a working Fiji/ImageJ installation (https://imagej.net/software/fiji)
- Fiji has a wide variety of supported operating systems (Windows XP, Vista, 7, 8, 10, 11, etc., Mac OS X 10.8 “Mountain Lion” or later, Linux on amd64 and x86 architectures)

## Installation guide
- If not already installed, download Fiji from the link above and run the executable.
- The macro itself does not need to be installed

## Instructions
1. Drag and drop an image in TIF (tif.) file to the Fiji toolbar. 
2. Drag and drop the Vesselcounting macro （.ijm） to the Fiji toobar.
3. Click the Run button in the Vesselcounting.ijm window.
4. Select the 'Oval' tool to select the whole xylem region.
5. Press 'OK' in the dialog box 'Please select ROI'
6. Press ROI>More>Labels, in this step all the vessels automatically recognized by this macro were labelled with a number.
7. Manually check to delete the false positive selection, and by pressing T shortcut key to add false negative selection with the help of Wand Tool.  (Wand Tool is setted with Tolerance in 20).
8. Update the Results file by clearing automatically generated results, and pressing ROI>Meausure to update results.
9. Save Results file.

## Testing image
You can test the macro with the attached tif image
