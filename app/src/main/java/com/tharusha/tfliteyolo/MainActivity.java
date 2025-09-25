package com.tharusha.tfliteyolo;

// Các import cần thiết cho xử lý ảnh, camera, giao diện...
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.*;
import android.os.Bundle;

import com.google.android.material.snackbar.Snackbar;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.view.Surface;
import android.view.TextureView;
import android.view.View;

import androidx.core.app.ActivityCompat;

import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;

import java.util.Arrays;
import java.util.List;

// Lớp MainActivity là activity chính điều khiển giao diện và xử lý camera + AI
public class MainActivity extends AppCompatActivity {
    private static final int CAMERA_PERMISSION_REQUEST_CODE = 100; // Mã yêu cầu quyền camera

    // Các thành phần giao diện
    private LinearLayout layoutStart;
    private Button buttonStart;
    private RelativeLayout layoutDetect;
    private Button buttonDetect;
    private TextureView textureViewCamera;     // Hiển thị luồng camera
    private ImageView imageViewDetection;      // Hiển thị ảnh có bounding box

    // Biến xử lý camera
    private CameraDevice cameraDevice;
    private CameraCaptureSession cameraCaptureSession;

    // Biến xử lý mô hình YOLO
    private YOLOModel yoloModel;
    private YOLODetector yoloDetector;

    // Hàm được gọi khi activity được khởi tạo
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Ánh xạ các view trong layout XML
        layoutStart = findViewById(R.id.linearLayout_start);
        buttonStart = findViewById(R.id.button_start);
        layoutDetect = findViewById(R.id.relativeLayout_detect);
        buttonDetect = findViewById(R.id.button_detect);
        textureViewCamera = findViewById(R.id.textureView_camera);
        imageViewDetection = findViewById(R.id.imageView_detection);

        // Xử lý khi nhấn nút Start
        buttonStart.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                start(); // Bắt đầu camera và load model
            }
        });

        // Xử lý khi nhấn nút Detect / Cancel
        buttonDetect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                switch (buttonDetect.getText().toString()) {
                    case "Detect":
                        runDetectObjectsThread(); // Bắt đầu luồng phát hiện vật
                        break;
                    case "Cancel":
                        setVisibleCameraStream(); // Quay lại stream camera
                        break;
                }
            }
        });
    }

    // Xử lý kết quả khi người dùng cho phép / từ chối quyền camera
    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                start(); // Nếu cấp quyền, bắt đầu
            } else {
                Snackbar.make(findViewById(R.id.button_start), "Camera permission denied!", Snackbar.LENGTH_LONG)
                        .setAction("Action", null).show();
            }
        }
    }

    // Hàm bắt đầu camera và mô hình
    void start() {
        if (!checkCameraPermission()) {
            requestCameraPermission(); // Yêu cầu cấp quyền nếu chưa có
            return;
        }

        // Hiển thị layout phát hiện, ẩn layout bắt đầu
        layoutStart.setVisibility(View.GONE);
        layoutDetect.setVisibility(View.VISIBLE);

        startCameraStream(); // Bắt đầu stream camera
        loadModel();         // Load mô hình YOLO
    }

    // Thiết lập TextureView để hiển thị camera
    void startCameraStream() {
        if (!checkCameraPermission()) {
            requestCameraPermission();
            return;
        }

        // Thiết lập listener cho TextureView (camera preview)
        textureViewCamera.setSurfaceTextureListener(new TextureView.SurfaceTextureListener() {
            @Override
            public void onSurfaceTextureAvailable(@NonNull SurfaceTexture surface, int width, int height) {
                openCamera(); // Khi Surface sẵn sàng, mở camera
            }

            @Override public void onSurfaceTextureSizeChanged(@NonNull SurfaceTexture surface, int width, int height) {}
            @Override public boolean onSurfaceTextureDestroyed(@NonNull SurfaceTexture surface) { return true; }
            @Override public void onSurfaceTextureUpdated(@NonNull SurfaceTexture surface) {}
        });
    }

    // Mở camera sử dụng Camera2 API
    void openCamera() {
        if (!checkCameraPermission()) return;

        CameraManager cameraManager = (CameraManager) getSystemService(CAMERA_SERVICE);

        try {
            String cameraId = cameraManager.getCameraIdList()[0]; // Dùng camera đầu tiên (thường là camera sau)
            cameraManager.openCamera(cameraId, new CameraDevice.StateCallback() {
                @Override public void onOpened(@NonNull CameraDevice camera) {
                    cameraDevice = camera;
                    createCameraPreviewSession(); // Khi camera mở thành công, tạo phiên preview
                }

                @Override public void onDisconnected(@NonNull CameraDevice camera) {
                    camera.close();
                    cameraDevice = null;
                }

                @Override public void onError(@NonNull CameraDevice camera, int error) {
                    camera.close();
                    cameraDevice = null;
                }
            }, null);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Tạo phiên camera preview để stream hình ảnh lên TextureView
    void createCameraPreviewSession() {
        imageViewDetection.setVisibility(View.INVISIBLE);
        textureViewCamera.setVisibility(View.VISIBLE);

        try {
            SurfaceTexture surfaceTexture = textureViewCamera.getSurfaceTexture();
            surfaceTexture.setDefaultBufferSize(640, 640); // Đặt kích thước buffer preview
            Surface surface = new Surface(surfaceTexture);

            final CaptureRequest.Builder captureRequestBuilder =
                    cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            captureRequestBuilder.addTarget(surface);

            cameraDevice.createCaptureSession(Arrays.asList(surface),
                    new CameraCaptureSession.StateCallback() {
                        @Override
                        public void onConfigured(@NonNull CameraCaptureSession session) {
                            if (cameraDevice == null) return;

                            cameraCaptureSession = session;
                            try {
                                captureRequestBuilder.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO);
                                cameraCaptureSession.setRepeatingRequest(captureRequestBuilder.build(), null, null);
                            } catch (CameraAccessException e) {
                                e.printStackTrace();
                            }
                        }

                        @Override
                        public void onConfigureFailed(@NonNull CameraCaptureSession session) {}
                    }, null);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Chạy luồng phát hiện vật (phát hiện trong background thread)
    void runDetectObjectsThread() {
        buttonDetect.setText("Please wait...");
        buttonDetect.setActivated(false);

        new Thread(new Runnable() {
            @Override
            public void run() {
                detectObjects();
            }
        }).start();
    }

    // Hàm chính thực hiện việc nhận diện vật thể
    void detectObjects() {
        Bitmap image = textureViewCamera.getBitmap(); // Lấy frame hiện tại từ camera

        // Vẽ ảnh ban đầu lên UI (trước khi có bounding box)
        runOnUiThread(() -> drawDetectionBitmap(image));

        if (image == null) {
            Snackbar.make(findViewById(R.id.button_detect), "Error detecting objects. Image is null!", Snackbar.LENGTH_LONG)
                    .setAction("Action", null).show();
            return;
        }

        // Phát hiện vật thể bằng YOLO
        List<YOLODetection> detections = yoloDetector.detectObjects(image);

        // Vẽ bounding box lên ảnh
        Bitmap outputImage = drawBoundingBox(image, detections);

        // Cập nhật UI với ảnh có bounding box
        runOnUiThread(() -> {
            drawDetectionBitmap(outputImage);
            buttonDetect.setText("Cancel");
            buttonDetect.setActivated(true);
        });

        Snackbar.make(findViewById(R.id.button_detect), "Objects detected", Snackbar.LENGTH_LONG)
                .setAction("Action", null).show();
    }

    // Hàm vẽ bounding box và label lên ảnh
    public Bitmap drawBoundingBox(Bitmap bitmap, List<YOLODetection> detections) {
        System.out.println("Num detections: " + detections.size());

        // Tạo bản sao của ảnh gốc để vẽ lên
        Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(mutableBitmap);

        Paint paint = new Paint();
        paint.setColor(Color.GREEN);
        paint.setStyle(Paint.Style.STROKE);

        for (YOLODetection detection : detections) {
            float x = detection.box_x;
            float y = detection.box_y;
            float width = detection.box_width;
            float height = detection.box_height;

            // Tính toạ độ góc bounding box
            float left = x - width / 2;
            float top = y - height / 2;
            float right = x + width / 2;
            float bottom = y + height / 2;

            paint.setStrokeWidth(10);
            canvas.drawRect(new RectF(left, top, right, bottom), paint);

            // Ghi label và độ chính xác
            String label = detection.className + " (" + String.format("%.2f", detection.confidence * 100) + "%)";
            paint.setStrokeWidth(2);
            paint.setTextSize(20);
            canvas.drawText(label, left, top - 10, paint);

            System.out.println("ClassIndex: " + detection.classIndex + ", ClassName: " + detection.className + ", Confidence: " + detection.confidence);
        }

        return mutableBitmap;
    }

    // Load mô hình YOLO và khởi tạo detector
    void loadModel() {
        yoloModel = new YOLOModel(this, Constants.MODEL_PATH, Constants.CLASSES_PATH);
        yoloDetector = new YOLODetector(yoloModel);

        Snackbar.make(findViewById(R.id.button_start), "Model loaded", Snackbar.LENGTH_LONG)
                .setAction("Action", null).show();
    }

    // Yêu cầu quyền camera
    void requestCameraPermission() {
        ActivityCompat.requestPermissions(this, new String[]{android.Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST_CODE);
    }

    // Kiểm tra xem có quyền dùng camera chưa
    boolean checkCameraPermission() {
        return ActivityCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }

    // Hiển thị lại stream camera sau khi nhấn Cancel
    void setVisibleCameraStream() {
        textureViewCamera.setVisibility(View.VISIBLE);
        imageViewDetection.setVisibility(View.INVISIBLE);
        buttonDetect.setText("Detect");
    }

    // Hiển thị ảnh đã vẽ bounding box
    void drawDetectionBitmap(Bitmap bitmap) {
        textureViewCamera.setVisibility(View.INVISIBLE);
        imageViewDetection.setVisibility(View.VISIBLE);
        imageViewDetection.setImageBitmap(bitmap);
    }
}
