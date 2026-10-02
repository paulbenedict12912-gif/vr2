package com.example.handscannervr

import android.Manifest
import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PixelFormat
import android.graphics.PorterDuff
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.util.Log
import android.util.Size
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.toBitmap
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import java.util.concurrent.Executors
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

private const val TAG = "HandScannerVR"

private val HAND_CONNECTIONS = arrayOf(
    intArrayOf(0,1), intArrayOf(1,2), intArrayOf(2,3), intArrayOf(3,4),
    intArrayOf(0,5), intArrayOf(5,6), intArrayOf(6,7), intArrayOf(7,8),
    intArrayOf(5,9), intArrayOf(9,10), intArrayOf(10,11), intArrayOf(11,12),
    intArrayOf(9,13), intArrayOf(13,14), intArrayOf(14,15), intArrayOf(15,16),
    intArrayOf(13,17), intArrayOf(17,18), intArrayOf(18,19), intArrayOf(19,20),
    intArrayOf(0,17)
)

private val FINGERTIPS = intArrayOf(4, 8, 12, 16, 20)

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: HandOverlayView
    private lateinit var statusText: TextView
    private lateinit var startOverlayBtn: Button
    private lateinit var accessibilityBtn: Button

    private val executor = Executors.newSingleThreadExecutor()
    private var landmarker: HandLandmarker? = null
    private var lensFacing = CameraSelector.LENS_FACING_FRONT
    private var imgW = 1
    private var imgH = 1

    private val askCamera =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera()
            else {
                Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
                finish()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        buildUi()

        if (!setupModel()) {
            Toast.makeText(this, "Could not load hand model", Toast.LENGTH_LONG).show()
            finish()
            return
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) startCamera()
        else askCamera.launch(Manifest.permission.CAMERA)
    }

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    private fun buildUi() {
        val root = FrameLayout(this).apply { setBackgroundColor(Color.BLACK) }

        previewView = PreviewView(this)
        root.addView(previewView, FrameLayout.LayoutParams(-1, -1))

        overlayView = HandOverlayView(this)
        root.addView(overlayView, FrameLayout.LayoutParams(-1, -1))

        val statusRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val dot = View(this).apply {
            layoutParams = LinearLayout.LayoutParams(dp(9), dp(9))
            background = GradientDrawable().apply {
                shape = GradientDrawable.OVAL
                setColor(0xFFFF4D5E.toInt())
            }
        }
        statusText = TextView(this).apply {
            text = "SEARCHING…"
            setTextColor(0xFF00F0FF.toInt())
            textSize = 12f
            letterSpacing = 0.18f
            typeface = Typeface.MONOSPACE
            setPadding(dp(10), 0, 0, 0)
        }
        statusRow.addView(dot)
        statusRow.addView(statusText)

        root.addView(statusRow, FrameLayout.LayoutParams(-2, -2).apply {
            gravity = Gravity.TOP or Gravity.START
            setMargins(dp(20), dp(30), 0, 0)
        })

        val btnRow = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL
        }

        accessibilityBtn = Button(this).apply {
            text = "ENABLE CLICKER"
            setOnClickListener { openAccessibilitySettings() }
        }
        startOverlayBtn = Button(this).apply {
            text = "START VR OVERLAY"
            setOnClickListener { startOverlay() }
        }

        btnRow.addView(accessibilityBtn)
        btnRow.addView(startOverlayBtn)

        root.addView(btnRow, FrameLayout.LayoutParams(-2, -2).apply {
            gravity = Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL
            setMargins(0, 0, 0, dp(40))
        })

        setContentView(root)
    }

    // ------------------------------------------------------------- model

    private fun setupModel(): Boolean {
        landmarker = try {
            createLandmarker(BaseOptions.Delegate.GPU)
        } catch (e: Throwable) {
            Log.w(TAG, "GPU failed, trying CPU", e)
            try { createLandmarker(BaseOptions.Delegate.CPU) }
            catch (e2: Throwable) { Log.e(TAG, "model init failed", e2); null }
        }
        return landmarker != null
    }

    private fun createLandmarker(delegate: BaseOptions.Delegate): HandLandmarker {
        val base = BaseOptions.builder()
            .setModelAssetPath("hand_landmarker.task")
            .setDelegate(delegate)
            .build()

        val options = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(base)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setNumHands(1)
            .setMinHandDetectionConfidence(0.5f)
            .setMinHandPresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .setResultListener { result, _ -> onResult(result) }
            .setErrorListener { e -> Log.e(TAG, "landmarker error", e) }
            .build()

        return HandLandmarker.createFromOptions(this, options)
    }

    // ------------------------------------------------------------- camera

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            @Suppress("DEPRECATION")
            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            analysis.setAnalyzer(executor) { proxy -> analyze(proxy) }

            val selector = CameraSelector.Builder()
                .requireLensFacing(lensFacing).build()

            try {
                provider.unbindAll()
                provider.bindToLifecycle(this, selector, preview, analysis)
            } catch (e: Exception) {
                Log.e(TAG, "bind failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyze(proxy: ImageProxy) {
        val lm = landmarker ?: run { proxy.close(); return }
        try {
            val bmp = proxy.toBitmap()
            if (bmp.width != imgW || bmp.height != imgH) {
                imgW = bmp.width; imgH = bmp.height
                runOnUiThread { overlayView.setImageSize(imgW, imgH) }
            }
            lm.detectAsync(BitmapImageBuilder(bmp).build(), System.currentTimeMillis())
        } catch (e: Throwable) {
            Log.e(TAG, "analyze failed", e)
        } finally {
            proxy.close()
        }
    }

    private fun onResult(result: HandLandmarkerResult) {
        val mirror = lensFacing == CameraSelector.LENS_FACING_FRONT
        runOnUiThread {
            overlayView.setResults(result, mirror)
            val n = result.landmarks().size
            statusText.text = when {
                n == 0 -> "SEARCHING…"
                else   -> "HAND LOCKED"
            }
            val dot = (statusText.parent as? LinearLayout)?.getChildAt(0) as? View
            dot?.background = GradientDrawable().apply {
                shape = GradientDrawable.OVAL
                setColor(if (n > 0) 0xFF00F0FF.toInt() else 0xFFFF4D5E.toInt())
            }
        }
    }

    // -------------------------------------------------- accessibility

    private fun openAccessibilitySettings() {
        Toast.makeText(this, "Find 'Hand Scanner VR Clicker' and turn it ON", Toast.LENGTH_LONG).show()
        startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
    }

    private fun startOverlay() {
        if (!Settings.canDrawOverlays(this)) {
            val intent = Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                Uri.parse("package:$packageName"))
            startActivity(intent)
            Toast.makeText(this, "Allow overlay permission, then tap START again", Toast.LENGTH_LONG).show()
            return
        }
        startService(Intent(this, OverlayService::class.java))
        Toast.makeText(this, "Overlay started. Switch to Roblox!", Toast.LENGTH_SHORT).show()
        moveTaskToBack(true)
    }

    override fun onDestroy() {
        super.onDestroy()
        executor.shutdown()
        landmarker?.close()
    }
}

/* ================================================================== *
 *  Accessibility Service — actually performs the screen tap
 * ================================================================== */

class HandAccessibilityService : AccessibilityService() {

    companion object {
        var instance: HandAccessibilityService? = null
        private var lastClickTime = 0L

        fun clickAt(x: Float, y: Float) {
            val now = System.currentTimeMillis()
            if (now - lastClickTime < 400) return  // debounce
            lastClickTime = now

            val svc = instance ?: return
            val path = Path().apply { moveTo(x, y) }
            val gesture = GestureDescription.Builder()
                .addStroke(GestureDescription.StrokeDescription(path, 0, 60))
                .build()
            svc.dispatchGesture(gesture, null, null)
        }
    }

    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
    }

    override fun onAccessibilityEvent(event: android.view.accessibility.AccessibilityEvent?) {}

    override fun onInterrupt() {}

    override fun onDestroy() {
        super.onDestroy()
        if (instance == this) instance = null
    }
}

/* ================================================================== *
 *  Overlay Service — draws the skeleton on top of any app
 * ================================================================== */

class OverlayService : android.app.Service() {

    private var overlayView: HandOverlayView? = null
    private var windowManager: WindowManager? = null

    override fun onBind(intent: Intent?) = null

    override fun onCreate() {
        super.onCreate()
        windowManager = getSystemService(WINDOW_SERVICE) as WindowManager
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (overlayView == null) {
            val params = WindowManager.LayoutParams(
                WindowManager.LayoutParams.MATCH_PARENT,
                WindowManager.LayoutParams.MATCH_PARENT,
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
                    WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
                else
                    WindowManager.LayoutParams.TYPE_PHONE,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
                        or WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE
                        or WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
                PixelFormat.TRANSLUCENT
            )
            overlayView = HandOverlayView(this)
            windowManager?.addView(overlayView, params)
        }
        return START_STICKY
    }

    override fun onDestroy() {
        super.onDestroy()
        overlayView?.let { windowManager?.removeView(it) }
        overlayView = null
    }
}

/* ================================================================== *
 *  Overlay view — draws the glowing hand skeleton + triggers clicks
 * ================================================================== */

class HandOverlayView(context: Context) : View(context) {

    private var hands: List<List<NormalizedLandmark>> = emptyList()
    private var srcW = 1
    private var srcH = 1
    private var mirror = true
    private val t0 = System.currentTimeMillis()
    private var lastPinchState = false

    private val glow = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
        color = 0x6600F0FF.toInt()
    }
    private val core = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
        color = 0xFFE8FFFF.toInt()
    }
    private val joint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = 0xFFEAFFFF.toInt()
    }
    private val ring = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = 0xFF7B5CFF.toInt()
    }
    private val bracket = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        color = 0xCC00F0FF.toInt()
    }
    private val cursorPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = 0xFFFF00AA.toInt()
        strokeWidth = 4f
    }
    private val path = Path()

    fun setImageSize(w: Int, h: Int) {
        if (w == srcW && h == srcH) return
        srcW = w; srcH = h; invalidate()
    }

    fun setResults(r: HandLandmarkerResult, mirrorHorizontally: Boolean) {
        hands = r.landmarks()
        mirror = mirrorHorizontally
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)

        if (hands.isEmpty()) return
        val vw = width.toFloat()
        val vh = height.toFloat()
        if (vw <= 0f || vh <= 0f) return

        val t = (System.currentTimeMillis() - t0) / 1000f
        val pulse = 1f + 0.22f * sin(t * 5f)

        val scale = max(vw / srcW, vh / srcH)
        val dw = srcW * scale
        val dh = srcH * scale
        val ox = (vw - dw) / 2f
        val oy = (vh - dh) / 2f

        val lineW = max(3f, vw * 0.0055f)
        val jointR = max(3f, vw * 0.007f)
        glow.strokeWidth = lineW * 2.8f
        core.strokeWidth = lineW
        ring.strokeWidth = max(1.5f, vw * 0.0035f)
        bracket.strokeWidth = max(2f, vw * 0.004f)

        for (hand in hands) {
            if (hand.size < 21) continue

            val xs = FloatArray(21)
            val ys = FloatArray(21)
            for (i in 0 until 21) {
                val lm = hand[i]
                val px = ox + lm.x() * dw
                xs[i] = if (mirror) vw - px else px
                ys[i] = oy + lm.y() * dh
            }

            path.reset()
            for (c in HAND_CONNECTIONS) {
                path.moveTo(xs[c[0]], ys[c[0]])
                path.lineTo(xs[c[1]], ys[c[1]])
            }

            canvas.drawPath(path, glow)
            canvas.drawPath(path, core)

            for (i in 0 until 21) canvas.drawCircle(xs[i], ys[i], jointR, joint)
            for (i in FINGERTIPS)
                canvas.drawCircle(xs[i], ys[i], jointR * 2.8f * pulse, ring)

            // ---- cursor: follow index fingertip (landmark 8) ----
            val cursorX = xs[8]
            val cursorY = ys[8]
            canvas.drawCircle(cursorX, cursorY, 18f, cursorPaint)
            canvas.drawCircle(cursorX, cursorY, 4f, cursorPaint)

            // ---- pinch detection: distance between thumb tip (4) and index tip (8) ----
            val dx = xs[4] - xs[8]
            val dy = ys[4] - ys[8]
            val dist = hypot(dx, dy)
            val pinchNow = dist < 60f  // threshold in screen pixels

            if (pinchNow && !lastPinchState) {
                // Fire a click at the cursor position
                HandAccessibilityService.clickAt(cursorX, cursorY)
            }
            lastPinchState = pinchNow

            // ---- targeting brackets ----
            var minX = Float.MAX_VALUE; var minY = Float.MAX_VALUE
            var maxX = -Float.MAX_VALUE; var maxY = -Float.MAX_VALUE
            for (i in 0 until 21) {
                if (xs[i] < minX) minX = xs[i]
                if (xs[i] > maxX) maxX = xs[i]
                if (ys[i] < minY) minY = ys[i]
                if (ys[i] > maxY) maxY = ys[i]
            }
            val pad = vw * 0.06f
            drawBrackets(canvas, minX - pad, minY - pad, maxX + pad, maxY + pad)
        }
    }

    private fun drawBrackets(c: Canvas, x0: Float, y0: Float, x1: Float, y1: Float) {
        val len = min(x1 - x0, y1 - y0) * 0.24f
        path.reset()
        path.moveTo(x0, y0 + len); path.lineTo(x0, y0); path.lineTo(x0 + len, y0)
        path.moveTo(x1 - len, y0); path.lineTo(x1, y0); path.lineTo(x1, y0 + len)
        path.moveTo(x1, y1 - len); path.lineTo(x1, y1); path.lineTo(x1 - len, y1)
        path.moveTo(x0 + len, y1); path.lineTo(x0, y1); path.lineTo(x0, y1 - len)
        c.drawPath(path, bracket)
    }
}
