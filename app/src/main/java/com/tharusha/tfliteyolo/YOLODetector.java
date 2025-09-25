package com.tharusha.tfliteyolo;

import android.graphics.Bitmap;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.ops.CastOp;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

// Lớp YOLODetector dùng để xử lý ảnh đầu vào, chạy mô hình YOLO và trả về các đối tượng phát hiện được
public class YOLODetector {
    private final YOLOModel yoloModel; // Mô hình YOLO đã load
    private final int inputImageSize;  // Kích thước ảnh đầu vào mà mô hình yêu cầu (ví dụ: 224x224)

    private float imageWidthOriginal;  // Chiều rộng ảnh gốc
    private float imageHeightOriginal; // Chiều cao ảnh gốc

    // Khởi tạo bộ xử lý ảnh (image preprocessor)
    // NormalizeOp: chia pixel cho 255 → đưa về khoảng [0, 1]
    // CastOp: ép kiểu sang FLOAT32 (kiểu dữ liệu mà mô hình yêu cầu)
    private final ImageProcessor imageProcessor = new ImageProcessor.Builder()
            .add(new NormalizeOp(0f, 255f)) // Normalize pixel từ [0,255] -> [0,1]
            .add(new CastOp(DataType.FLOAT32)) // Ép kiểu dữ liệu sang float32
            .build();

    // Hàm khởi tạo YOLODetector, nhận vào đối tượng YOLOModel
    public YOLODetector(YOLOModel model) {
        this.yoloModel = model; // Gán model YOLO
        this.inputImageSize = model.inputShape[1]; // Lấy kích thước ảnh đầu vào từ model
    }

    // Hàm phát hiện đối tượng trong ảnh Bitmap
    public List<YOLODetection> detectObjects(Bitmap bitmap) {
        // Lưu lại kích thước ảnh gốc để scale bounding box sau này
        imageWidthOriginal = bitmap.getWidth();
        imageHeightOriginal = bitmap.getHeight();

        // Resize ảnh về đúng kích thước mà mô hình yêu cầu
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputImageSize, inputImageSize, false);

        // Tạo đối tượng TensorImage và load ảnh vào
        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
        tensorImage.load(resizedBitmap);

        // Áp dụng tiền xử lý ảnh: normalize + cast
        tensorImage = imageProcessor.process(tensorImage);

        // Tạo mảng đầu ra để chứa kết quả từ mô hình
        float[][][] output = new float[1][yoloModel.outputShape[1]][yoloModel.outputShape[2]];

        // Chạy mô hình YOLO với ảnh đầu vào đã xử lý
        yoloModel.interpreter.run(tensorImage.getBuffer(), output);

        // Xử lý kết quả đầu ra và trả về danh sách các đối tượng phát hiện được
        return processYOLOOutput(output);
    }

    // Hàm xử lý kết quả đầu ra của mô hình YOLO để tạo danh sách YOLODetection
    List<YOLODetection> processYOLOOutput(float[][][] yoloOutput) {
        List<YOLODetection> detections = new ArrayList<>();

        int numDetections = yoloModel.outputShape[2]; // Số lượng hộp phát hiện (detection boxes)
        int numClasses = yoloModel.outputShape[1] - 4; // Số lượng lớp (channels trừ đi 4: x, y, w, h)

        // Duyệt qua từng detection
        for (int i = 0; i < numDetections; i++) {
            int bestClass = -1;
            float bestClassScore = 0;

            // Tìm lớp có xác suất cao nhất trong detection này
            for (int j = 4; j < 4 + numClasses; j++) {
                if (yoloOutput[0][j][i] > bestClassScore) {
                    bestClassScore = yoloOutput[0][j][i];
                    bestClass = j - 4; // Lấy chỉ số lớp
                }
            }

            // Bỏ qua nếu xác suất thấp (confidence < 0.5)
            if (bestClassScore < 0.5) {
                continue;
            }

            // Tạo đối tượng YOLODetection
            YOLODetection detection = new YOLODetection();
            detection.classIndex = bestClass;
            detection.confidence = bestClassScore;

            // Lấy toạ độ bounding box từ kết quả và scale theo kích thước ảnh gốc
            detection.box_x = yoloOutput[0][0][i] * imageWidthOriginal;
            detection.box_y = yoloOutput[0][1][i] * imageHeightOriginal;
            detection.box_width = yoloOutput[0][2][i] * imageWidthOriginal;
            detection.box_height = yoloOutput[0][3][i] * imageHeightOriginal;

            // Gán tên lớp nếu có trong danh sách
            if (yoloModel.classes.size() > bestClass) {
                detection.className = yoloModel.classes.get(bestClass);
            } else {
                detection.className = "";
            }

            detections.add(detection);
        }

        // Loại bỏ các bounding box bị chồng lắp bằng NMS (Non-Max Suppression)
        return applyNMS(detections, 0.5f);
    }

    // Hàm thực hiện Non-Maximum Suppression (NMS) để loại bỏ các box chồng nhau
    private List<YOLODetection> applyNMS(List<YOLODetection> detections, float iouThreshold) {
        // Sắp xếp các detection theo độ tin cậy giảm dần
        Collections.sort(detections, new Comparator<YOLODetection>() {
            @Override
            public int compare(YOLODetection d1, YOLODetection d2) {
                return Float.compare(d2.confidence, d1.confidence);
            }
        });

        List<YOLODetection> finalDetections = new ArrayList<>();

        // Lặp qua từng detection và loại bỏ các box có IoU cao (chồng lắp)
        while (!detections.isEmpty()) {
            YOLODetection bestDetection = detections.remove(0);
            finalDetections.add(bestDetection);

            // Loại bỏ các box có IoU lớn hơn ngưỡng
            detections.removeIf(d -> computeIoU(bestDetection, d) > iouThreshold);
        }

        return finalDetections;
    }

    // Hàm tính IoU (Intersection over Union) giữa hai bounding box
    private float computeIoU(YOLODetection box1, YOLODetection box2) {
        // Tính toạ độ phần giao nhau
        float x1 = Math.max(box1.box_x, box2.box_x);
        float y1 = Math.max(box1.box_y, box2.box_y);
        float x2 = Math.min(box1.box_x + box1.box_width, box2.box_x + box2.box_width);
        float y2 = Math.min(box1.box_y + box1.box_height, box2.box_y + box2.box_height);

        // Diện tích phần giao nhau
        float intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);

        // Diện tích từng box
        float box1Area = box1.box_width * box1.box_height;
        float box2Area = box2.box_width * box2.box_height;

        // Tính IoU theo công thức: IoU = Intersection / Union
        float union = box1Area + box2Area - intersection;
        return union > 0 ? intersection / union : 0;
    }
}


