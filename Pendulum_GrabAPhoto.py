#coding=utf-8

import mvsdk
import numpy as np
import cv2
import time
import os
from datetime import datetime

def main():
    # Create folder for saving images
    save_dir = "Pendulumsofttri_camera_images"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Generate timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Enumerate cameras
    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    if nDev < 1:
        print("No camera was found!")
        return
        
    for i, DevInfo in enumerate(DevList):
        print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
    i = 0 if nDev == 1 else int(input("Select camera: "))
    DevInfo = DevList[i]

    # Open camera
    hCamera = 0
    try:
        hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
    except mvsdk.CameraException as e:
        print("CameraInit Failed({}): {}".format(e.error_code, e.message))
        return

    # Get camera capability description
    cap = mvsdk.CameraGetCapability(hCamera)

    # Set ISP output format to MONO8
    mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
    
    # Switch camera mode to continuous acquisition
    mvsdk.CameraSetTriggerMode(hCamera, 0)

    # Manual exposure settings
    mvsdk.CameraSetAeState(hCamera, 0)
    mvsdk.CameraSetExposureTime(hCamera, 0.8 * 1000)  # Exposure time: 0.8 ms
    mvsdk.CameraSetAnalogGain(hCamera, 64)           # Analog gain
    mvsdk.CameraSetGamma(hCamera, 250)               # Gamma
    mvsdk.CameraSetContrast(hCamera, 200)            # Contrast
    mvsdk.CameraSetAeTarget(hCamera, 10000)          # Brightness

    # Set resolution to 320x240
    resolution = mvsdk.CameraGetImageResolution(hCamera)
    resolution.iIndex = 1  # Use index 1 for 320x240 resolution
    mvsdk.CameraSetImageResolution(hCamera, resolution)

    # Set high frame rate mode
    mvsdk.CameraSetFrameSpeed(hCamera, 2)           # 2: high-speed mode

    # Start the SDK internal frame acquisition thread
    mvsdk.CameraPlay(hCamera)

    # Compute buffer size and allocate buffer
    FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax
    pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)
    
    # ROI parameters
    ## 11-samll tube
    # roi_x = 156 # ROI start x coordinate
    # roi_y = 19   # ROI start y coordinate
    # roi_width = 82  # ROI width
    # roi_height = 132 # Corrected to 123, consistent with reference code
    
    #Pendulum ROI
    roi_x = 86      # ROI start x coordinate
    roi_y = 104       # ROI start y coordinate
    roi_width = 132  # ROI width
    roi_height = 82  # Corrected to 123, consistent with reference code
    
    try:
        # Capture one frame
        pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 1000)
        mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
        mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)
        
        # Convert image data to OpenCV format
        frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
        frame = np.frombuffer(frame_data, dtype=np.uint8)
        frame = frame.reshape(FrameHead.iHeight, FrameHead.iWidth)

        # Save original image
        original_path = os.path.join(save_dir, f"original_{timestamp}.png")
        cv2.imwrite(original_path, frame)
        print(f"original pic saved to: {original_path}")

        # ROI crop
        roi = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
        
        # Inverse binary threshold
        _, binary = cv2.threshold(roi, 128, 255, cv2.THRESH_BINARY_INV)
        
        # Save binary image
        binary_path = os.path.join(save_dir, f"binary_{timestamp}.png")
        cv2.imwrite(binary_path, binary)
        print(f"binary pic saved to: {binary_path}")
        
        # Remove connected components smaller than 500 pixels
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        filtered_binary = np.zeros_like(binary)
        
        # Draw centroids on ROI
        roi_display = cv2.cvtColor(roi.copy(), cv2.COLOR_GRAY2BGR)
        
        # Also draw markers on the original image (for saving the full marked image)
        frame_marked = cv2.cvtColor(frame.copy(), cv2.COLOR_GRAY2BGR)
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 500:
                filtered_binary[labels == i] = 255
                centroid_x = centroids[i][0]
                centroid_y = centroids[i][1]
                pixel_x = centroid_x + roi_x  # Convert to original image coordinate system
                pixel_y = centroid_y + roi_y  # Convert to original image coordinate system
                
                # Draw centroid position on ROI
                cv2.circle(roi_display, (int(centroid_x), int(centroid_y)), 3, (0, 0, 255), -1)
                # cv2.putText(roi_display, f"({centroid_x:.0f},{centroid_y:.0f})", 
                #            (int(centroid_x)+5, int(centroid_y)-5), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # Draw ROI box and centroid on original image
                cv2.rectangle(frame_marked, (roi_x, roi_y), 
                            (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)
                cv2.circle(frame_marked, (int(pixel_x), int(pixel_y)), 5, (0, 0, 255), -1)
                
                # Manually draw a cross marker (instead of cv2.cross)
                cross_size = 10
                cv2.line(frame_marked, 
                        (int(pixel_x) - cross_size, int(pixel_y)), 
                        (int(pixel_x) + cross_size, int(pixel_y)), 
                        (255, 0, 0), 1)
                cv2.line(frame_marked, 
                        (int(pixel_x), int(pixel_y) - cross_size), 
                        (int(pixel_x), int(pixel_y) + cross_size), 
                        (255, 0, 0), 1)
                
                # cv2.putText(frame_marked, f"({pixel_x:.0f},{pixel_y:.0f})", 
                #            (int(pixel_x)+8, int(pixel_y)-8), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Print coordinate information
                print(f"Centroided pixel cor- X: {pixel_x:.1f}, Y: {pixel_y:.1f}")

            # Save ROI image with centroid markers
        roi_display_path = os.path.join(save_dir, f"roi_with_centroids_{timestamp}.png")
        cv2.imwrite(roi_display_path, roi_display)
        print(f"ROI with centroids pic saved to: {roi_display_path}")

        # Save marked image
        marked_path = os.path.join(save_dir, f"marked_{timestamp}.png")
        cv2.imwrite(marked_path, frame_marked)
        print(f"marked pic saved to: {marked_path}")
        
        # Save filtered binary image
        filtered_binary_path = os.path.join(save_dir, f"filtered_binary_{timestamp}.png")
        cv2.imwrite(filtered_binary_path, filtered_binary)
        print(f"filtered_binary pic saved to: {filtered_binary_path}")

        # Display processing results
        cv2.imshow("Original", frame)
        cv2.imshow("Binary", binary)
        cv2.imshow("Filtered Binary", filtered_binary)
        cv2.imshow("ROI with Centroids", roi_display)
        cv2.imshow("Marked Image", frame_marked)
        
        print(f"\nAll pics saved to: {save_dir}")
        print("Summary of saved images:")
        print(f"  1. original_{timestamp}.png - Original grayscale image")
        print(f"  2. binary_{timestamp}.png - Binary thresholded ROI")
        print(f"  3. roi_with_centroids_{timestamp}.png - ROI with centroid markers")
        print(f"  4. marked_{timestamp}.png - Full image with ROI box and centroids")
        print(f"  5. filtered_binary_{timestamp}.png - Filtered binary (areas >= 500px)")
        
        # Wait for key press
        cv2.waitKey(0)

    except mvsdk.CameraException as e:
        print("Camera error({}): {}".format(e.error_code, e.message))

    # Clean up resources
    cv2.destroyAllWindows()
    mvsdk.CameraStop(hCamera)
    mvsdk.CameraUnInit(hCamera)
    mvsdk.CameraAlignFree(pFrameBuffer)

if __name__ == '__main__':
    main()