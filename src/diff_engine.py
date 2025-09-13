import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage


class DifferenceEngine:
    def __init__(self, sensitivity=0.3):
        self.sensitivity = sensitivity
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()

    def compute_difference(self, frame1, frame2):
        """
        Multi-method difference detection for robustness
        """
        # Method 1: Structural Similarity Index
        diff_map_ssim = self._ssim_difference(frame1, frame2)

        # Method 2: Absolute difference with threshold
        diff_map_abs = self._absolute_difference(frame1, frame2)

        # Method 3: Background subtraction (for moving objects)
        diff_map_bg = self._background_subtraction(frame2)

        # Combine methods with weighted average
        combined_diff = self._combine_methods(
            diff_map_ssim,
            diff_map_abs,
            diff_map_bg
        )

        # Generate highlight regions
        highlights = self._generate_highlights(combined_diff)

        return combined_diff, highlights

    def _ssim_difference(self, frame1, frame2):
        """Structural similarity approach"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Compute SSIM
        score, diff = ssim(gray1, gray2, full=True)
        diff = (diff * 255).astype("uint8")

        # Invert (SSIM gives similarity, we want difference)
        diff = 255 - diff

        return diff

    def _absolute_difference(self, frame1, frame2):
        """Simple absolute difference with noise reduction"""
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
        gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

        # Compute absolute difference
        diff = cv2.absdiff(gray1, gray2)

        # Threshold to binary
        _, thresh = cv2.threshold(
            diff,
            int(255 * self.sensitivity),
            255,
            cv2.THRESH_BINARY
        )

        return thresh

    def _background_subtraction(self, frame):
        """MOG2 background subtraction for movement"""
        return self.background_subtractor.apply(frame)

    def _combine_methods(self, diff1, diff2, diff3):
        """Weighted combination of difference methods"""
        # Normalize all to same shape if needed
        h, w = diff1.shape[:2]
        diff2 = cv2.resize(diff2, (w, h))
        diff3 = cv2.resize(diff3, (w, h))

        # Weighted average (adjust weights based on testing)
        combined = (
            0.4 * diff1.astype(float) +
            0.3 * diff2.astype(float) +
            0.3 * diff3.astype(float)
        )

        return combined.astype(np.uint8)

    def _generate_highlights(self, diff_map):
        """Generate bounding boxes around changed regions"""
        # Threshold the combined diff_map to binary for contour detection
        _, binary_diff = cv2.threshold(
            diff_map,
            25,  # Slightly lower threshold for better detection
            255,
            cv2.THRESH_BINARY
        )

        # Apply larger morphological operations to merge nearby changes and create larger regions
        kernel_large = np.ones((15, 15), np.uint8)  # Larger kernel to connect nearby changes
        kernel_medium = np.ones((7, 7), np.uint8)   # Medium kernel for cleanup

        # First, close gaps to connect nearby change regions
        binary_diff = cv2.morphologyEx(binary_diff, cv2.MORPH_CLOSE, kernel_large)
        # Then clean up noise with opening
        binary_diff = cv2.morphologyEx(binary_diff, cv2.MORPH_OPEN, kernel_medium)
        # Final closing to ensure solid regions
        binary_diff = cv2.morphologyEx(binary_diff, cv2.MORPH_CLOSE, kernel_medium)

        # Find contours
        contours, _ = cv2.findContours(
            binary_diff,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Calculate minimum area threshold (0.5% of total video area)
        frame_height, frame_width = diff_map.shape[:2]
        total_frame_area = frame_width * frame_height
        min_area_threshold = total_frame_area * 0.005  # 0.5% of video area

        # For display boxes, keep minimum box size at 10% of dimensions for visibility
        min_box_width = int(frame_width * 0.1)
        min_box_height = int(frame_height * 0.1)

        # Filter and expand contours
        highlights = []
        total_contours = len(contours)
        filtered_count = 0

        for contour in contours:
            area = cv2.contourArea(contour)

            # Only process changes that meet the 5% area threshold
            if area < min_area_threshold:
                filtered_count += 1
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Expand small boxes to minimum display size for visibility
            expanded = False
            if w < min_box_width or h < min_box_height:
                # Calculate expansion needed
                expand_w = max(0, (min_box_width - w) // 2)
                expand_h = max(0, (min_box_height - h) // 2)

                # Apply expansion while staying within frame bounds
                x = max(0, x - expand_w)
                y = max(0, y - expand_h)
                w = min(frame_width - x, w + 2 * expand_w)
                h = min(frame_height - y, h + 2 * expand_h)
                expanded = True

            # Add to highlights (area requirement already met)
            highlights.append({
                'bbox': (x, y, w, h),
                'area': cv2.contourArea(contour),  # Use original contour area for reporting
                'display_area': w * h,  # Display box area (may be expanded)
                'center': (x + w//2, y + h//2),
                'contour': contour,
                'expanded': expanded
            })

        # Add metadata for visualization
        for highlight in highlights:
            highlight['_meta'] = {
                'total_contours': total_contours,
                'filtered_count': filtered_count,
                'min_area_threshold': min_area_threshold,
                'total_frame_area': total_frame_area
            }

        return highlights

    def visualize_difference(self, original_frame, diff_map, highlights):
        """Create visualization with highlights"""
        result = original_frame.copy()

        # Create a mask for significant changes (same threshold as highlights)
        _, change_mask = cv2.threshold(diff_map, 30, 255, cv2.THRESH_BINARY)

        # Apply color overlay only where there are significant changes
        diff_color = cv2.applyColorMap(diff_map, cv2.COLORMAP_HOT)

        # Create colored overlay only in change areas
        change_mask_3d = cv2.cvtColor(change_mask, cv2.COLOR_GRAY2BGR) / 255.0
        colored_changes = diff_color * change_mask_3d

        # Blend colored changes with original frame
        result = result.astype(float)
        result = result * (1 - change_mask_3d * 0.6) + colored_changes * 0.6
        result = result.astype(np.uint8)

        # Draw bounding boxes around major change regions
        for highlight in highlights:
            x, y, w, h = highlight['bbox']

            # Draw rectangle with thickness based on change area
            thickness = min(4, max(2, int(highlight['area'] / 50000)))

            # Use different color for expanded boxes
            color = (0, 255, 255) if highlight.get('expanded', False) else (0, 255, 0)  # Cyan for expanded, Green for original
            cv2.rectangle(result, (x, y), (x+w, y+h), color, thickness)

            # Add area indicator and expansion status
            actual_area = highlight['area']
            area_percent = (actual_area / (change_mask.shape[0] * change_mask.shape[1])) * 100
            area_text = f"{area_percent:.1f}% ({actual_area//1000}K px)"
            if highlight.get('expanded', False):
                area_text += " (expanded)"

            cv2.putText(
                result,
                area_text,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        # Add debug info
        change_pixels = np.sum(change_mask > 0)
        total_pixels = change_mask.shape[0] * change_mask.shape[1]
        change_percent = (change_pixels / total_pixels) * 100

        cv2.putText(
            result,
            f"Total Change: {change_percent:.1f}%",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Show filtering stats
        if highlights and '_meta' in highlights[0]:
            meta = highlights[0]['_meta']
            total_contours = meta['total_contours']
            min_area_threshold = meta['min_area_threshold']
            total_frame_area = meta['total_frame_area']

            if total_contours > 0:
                cv2.putText(
                    result,
                    f"Regions: {len(highlights)}/{total_contours} (>{min_area_threshold/total_frame_area*100:.1f}%)",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

        return result

    def get_boxes_only_frame(self, original_frame, highlights):
        """Create frame with only bounding boxes (no heatmap) for VLM analysis"""
        result = original_frame.copy()

        # Draw only bounding boxes (no colored overlay)
        for highlight in highlights:
            x, y, w, h = highlight['bbox']

            # Draw rectangle with consistent styling
            thickness = 3
            color = (0, 255, 0)  # Green boxes only
            cv2.rectangle(result, (x, y), (x+w, y+h), color, thickness)

            # Add simple label
            actual_area = highlight['area']
            area_percent = (actual_area / (original_frame.shape[0] * original_frame.shape[1])) * 100
            label = f"{area_percent:.1f}%"

            cv2.putText(
                result,
                label,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        return result

    def get_change_summary(self, highlights):
        """Generate a summary of detected changes"""
        if not highlights:
            return "No significant changes detected"

        total_area = sum(h['area'] for h in highlights)
        num_regions = len(highlights)

        # Sort by area to get largest changes
        highlights_sorted = sorted(highlights, key=lambda x: x['area'], reverse=True)

        summary = f"{num_regions} change region(s) detected"
        if num_regions > 0:
            largest = highlights_sorted[0]
            summary += f", largest area: {largest['area']:.0f} pixels"

        return summary