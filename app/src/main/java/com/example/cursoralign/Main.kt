package com.example.cursoralign

import android.app.Activity
import android.app.Application
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.PixelFormat
import android.graphics.Point
import android.graphics.Rect
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.Image
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.Process
import android.provider.Settings
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import rikka.shizuku.Shizuku
import java.nio.ByteBuffer
import kotlin.math.abs

class MainApplication : Application() {

    override fun onCreate() {
        super.onCreate()

        Prefs.init(this)

        try {
            OpenCVLoader.initLocal()
        } catch (_: Exception) {
        }
    }
}

object AppState {
    var enabled: Boolean = false
    var triggerActive: Boolean = false
    var targetPoint: Point? = null

    var screenW: Int = 0
    var screenH: Int = 0

    var cursorX: Int = 0
    var cursorY: Int = 0

    var floatingServiceRunning: Boolean = false
    var screenCaptureServiceRunning: Boolean = false
}

object Config {

    // Detection processing width.
    // Lower = faster, Higher = more accurate but slower.
    const val PROCESS_WIDTH = 640

    // Confidence thresholds.
    // 0.0 to 1.0.
    const val TARGET_THRESHOLD = 0.72
    const val ACTIVE_THRESHOLD = 0.75
    const val NORMAL_ACTIVE_MARGIN = 0.03

    // Cursor deadzone.
    const val DEADZONE = 24

    // Swipe control.
    const val SWIPE_INTERVAL_MS = 250L
    const val SWIPE_DURATION_MS = 120
    const val SWIPE_FACTOR = 0.55f

    // IMPORTANT:
    // If swipe direction is opposite, change this to false.
    const val SWIPE_MODE_AWAY_FROM_TARGET = true

    // If trigger button is fixed, set its screen area here.
    // Example:
    // val TRIGGER_ROI: Rect? = Rect(100, 1500, 350, 1750)
    val TRIGGER_ROI: Rect? = null
}

object Prefs {

    private const val PREF_NAME = "cursor_align_prefs"
    private const val KEY_ENABLED = "cursor_align_enabled"

    private lateinit var prefs: SharedPreferences

    fun init(context: Context) {
        prefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
    }

    fun isEnabled(): Boolean {
        return prefs.getBoolean(KEY_ENABLED, false)
    }

    fun setEnabled(value: Boolean) {
        prefs.edit().putBoolean(KEY_ENABLED, value).apply()
    }
}

object ShizukuInput {

    fun swipe(x1: Int, y1: Int, x2: Int, y2: Int, durationMs: Int) {
        if (!Shizuku.pingBinder()) return

        val command = "input swipe $x1 $y1 $x2 $y2 $durationMs"

        Thread {
            try {
                Shizuku.newProcess(arrayOf("sh", "-c", command), null, null).waitFor()
            } catch (_: Exception) {
            }
        }.start()
    }
}

data class MatchResult(
    val value: Double,
    val x: Int,
    val y: Int,
    val w: Int,
    val h: Int
)

data class DetectionResult(
    val active: Boolean,
    val target: Point?
)

class DetectionEngine(private val context: Context) {

    private val targetTemplates = mutableListOf<Mat>()
    private var triggerActiveTemplate: Mat? = null
    private var triggerNormalTemplate: Mat? = null

    private var loaded = false

    fun ensureLoaded() {
        if (loaded) return
        loadTemplates()
        loaded = true
    }

    private fun loadTemplates() {
        releaseTemplates()

        try {
            val files = context.assets.list("detection") ?: emptyArray()

            files
                .filter { it.startsWith("target_b") }
                .sorted()
                .forEach { fileName ->
                    val mat = loadGrayMat("detection/$fileName")
                    if (mat != null) {
                        targetTemplates.add(mat)
                    }
                }

            triggerActiveTemplate = loadGrayMat("detection/trigger_active.jpg")
            if (triggerActiveTemplate == null) {
                triggerActiveTemplate = loadGrayMat("detection/trigger_active.png")
            }

            triggerNormalTemplate = loadGrayMat("detection/trigger_normal.jpg")
            if (triggerNormalTemplate == null) {
                triggerNormalTemplate = loadGrayMat("detection/trigger_normal.png")
            }
        } catch (_: Exception) {
        }
    }

    private fun loadGrayMat(path: String): Mat? {
        return try {
            val bitmap = BitmapFactory.decodeStream(context.assets.open(path)) ?: return null

            val rgba = Mat()
            Utils.bitmapToMat(bitmap, rgba)

            val gray = Mat()

            if (rgba.channels() == 4) {
                Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)
            } else if (rgba.channels() == 3) {
                Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGB2GRAY)
            } else {
                rgba.copyTo(gray)
            }

            rgba.release()
            bitmap.recycle()

            gray
        } catch (_: Exception) {
            null
        }
    }

    fun process(bitmap: Bitmap, screenW: Int, screenH: Int): DetectionResult {
        ensureLoaded()

        if (bitmap.width <= 0 || bitmap.height <= 0) {
            return DetectionResult(false, null)
        }

        val rgba = Mat()
        Utils.bitmapToMat(bitmap, rgba)

        val scale = Config.PROCESS_WIDTH.toDouble() / screenW.toDouble()
        val processH = (screenH * scale).toInt().coerceAtLeast(1)

        val smallRgba = Mat()
        Imgproc.resize(
            rgba,
            smallRgba,
            Size(Config.PROCESS_WIDTH.toDouble(), processH.toDouble()),
            0.0,
            0.0,
            Imgproc.INTER_AREA
        )

        val gray = Mat()

        if (smallRgba.channels() == 4) {
            Imgproc.cvtColor(smallRgba, gray, Imgproc.COLOR_RGBA2GRAY)
        } else if (smallRgba.channels() == 3) {
            Imgproc.cvtColor(smallRgba, gray, Imgproc.COLOR_RGB2GRAY)
        } else {
            smallRgba.copyTo(gray)
        }

        rgba.release()
        smallRgba.release()

        val active = detectTriggerActive(gray, scale)

        var target: Point? = null

        if (AppState.enabled && active) {
            target = detectTarget(gray, scale)
        }

        gray.release()

        return DetectionResult(active, target)
    }

    private fun detectTriggerActive(gray: Mat, scale: Double): Boolean {
        val activeTemplate = triggerActiveTemplate ?: return false

        var searchMat = gray
        var subMat: Mat? = null

        Config.TRIGGER_ROI?.let { roiOriginal ->
            val left = (roiOriginal.left * scale).toInt().coerceIn(0, gray.cols() - 1)
            val top = (roiOriginal.top * scale).toInt().coerceIn(0, gray.rows() - 1)
            val right = (roiOriginal.right * scale).toInt().coerceIn(left + 1, gray.cols())
            val bottom = (roiOriginal.bottom * scale).toInt().coerceIn(top + 1, gray.rows())

            if (right - left > 20 && bottom - top > 20) {
                subMat = gray.submat(top, bottom, left, right)
                searchMat = subMat
            }
        }

        val activeMatch = matchTemplate(searchMat, activeTemplate, scale)

        var isActive = false

        if (activeMatch != null && activeMatch.value >= Config.ACTIVE_THRESHOLD) {
            isActive = true

            val normalTemplate = triggerNormalTemplate
            if (normalTemplate != null) {
                val normalMatch = matchTemplate(searchMat, normalTemplate, scale)
                if (normalMatch != null && normalMatch.value > activeMatch.value + Config.NORMAL_ACTIVE_MARGIN) {
                    isActive = false
                }
            }
        }

        subMat?.release()

        return isActive
    }

    private fun detectTarget(gray: Mat, scale: Double): Point? {
        var best: MatchResult? = null

        for (template in targetTemplates) {
            val match = matchTemplate(gray, template, scale)

            if (match != null && match.value >= Config.TARGET_THRESHOLD) {
                if (best == null || match.value > best.value) {
                    best = match
                }
            }
        }

        val result = best ?: return null

        val centerXSmall = result.x + result.w / 2.0
        val centerYSmall = result.y + result.h / 2.0

        val originalX = (centerXSmall / scale).toInt().coerceIn(0, AppState.screenW)
        val originalY = (centerYSmall / scale).toInt().coerceIn(0, AppState.screenH)

        return Point(originalX, originalY)
    }

    private fun matchTemplate(search: Mat, template: Mat, scale: Double): MatchResult? {
        if (search.empty() || template.empty()) return null

        val templateWidth = (template.cols() * scale).toInt()
        val templateHeight = (template.rows() * scale).toInt()

        if (templateWidth < 12 || templateHeight < 12) return null
        if (templateWidth >= search.cols() || templateHeight >= search.rows()) return null

        val resizedTemplate = Mat()
        Imgproc.resize(
            template,
            resizedTemplate,
            Size(templateWidth.toDouble(), templateHeight.toDouble()),
            0.0,
            0.0,
            Imgproc.INTER_AREA
        )

        val result = Mat()

        return try {
            Imgproc.matchTemplate(search, resizedTemplate, result, Imgproc.TM_CCOEFF_NORMED)
            val minMax = Core.minMaxLoc(result)

            MatchResult(
                value = minMax.maxVal,
                x = minMax.maxLoc.x.toInt(),
                y = minMax.maxLoc.y.toInt(),
                w = templateWidth,
                h = templateHeight
            )
        } catch (_: Exception) {
            null
        } finally {
            resizedTemplate.release()
            result.release()
        }
    }

    private fun releaseTemplates() {
        targetTemplates.forEach { it.release() }
        targetTemplates.clear()

        triggerActiveTemplate?.release()
        triggerNormalTemplate?.release()

        triggerActiveTemplate = null
        triggerNormalTemplate = null
    }
}

class MainActivity : Activity() {

    companion object {
        private const val OVERLAY_REQUEST_CODE = 1001
        private const val MEDIA_PROJECTION_REQUEST_CODE = 1002
        private const val NOTIFICATION_REQUEST_CODE = 1003
        private const val SHIZUKU_REQUEST_CODE = 1004
    }

    private lateinit var startStopButton: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val root = FrameLayout(this)

        startStopButton = Button(this)
        startStopButton.text = "Start"
        startStopButton.setOnClickListener {
            if (isRunning()) {
                stopFlow()
            } else {
                startFlow()
            }
        }

        val params = FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
            Gravity.CENTER
        )

        root.addView(startStopButton, params)
        setContentView(root)

        updateButton()
    }

    override fun onResume() {
        super.onResume()
        updateButton()
    }

    private fun isRunning(): Boolean {
        return AppState.floatingServiceRunning || AppState.screenCaptureServiceRunning
    }

    private fun updateButton() {
        startStopButton.text = if (isRunning()) "Stop" else "Start"
    }

    private fun startFlow() {
        if (!Settings.canDrawOverlays(this)) {
            requestOverlayPermission()
            return
        }

        if (!Shizuku.pingBinder()) {
            toast("Shizuku install karo aur Shizuku ko wireless debugging/ADB se start karo")
            return
        }

        if (Shizuku.checkSelfPermission() != PackageManager.PERMISSION_GRANTED) {
            Shizuku.requestPermission(SHIZUKU_REQUEST_CODE)
            return
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            val permission = android.Manifest.permission.POST_NOTIFICATIONS
            if (checkSelfPermission(permission) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(arrayOf(permission), NOTIFICATION_REQUEST_CODE)
                return
            }
        }

        requestScreenCapture()
    }

    private fun requestOverlayPermission() {
        toast("Display over other apps permission allow karo")

        val intent = Intent(
            Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
            android.net.Uri.parse("package:$packageName")
        )

        @Suppress("DEPRECATION")
        startActivityForResult(intent, OVERLAY_REQUEST_CODE)
    }

    private fun requestScreenCapture() {
        val mediaProjectionManager =
            getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager

        val intent = mediaProjectionManager.createScreenCaptureIntent()

        @Suppress("DEPRECATION")
        startActivityForResult(intent, MEDIA_PROJECTION_REQUEST_CODE)
    }

    private fun startCaptureService(data: Intent) {
        AppState.screenCaptureServiceRunning = true

        val intent = Intent(this, ScreenCaptureService::class.java)
        intent.putExtra(ScreenCaptureService.EXTRA_DATA, data)

        startForegroundService(intent)
    }

    private fun startFloatingService() {
        AppState.floatingServiceRunning = true

        val intent = Intent(this, FloatingService::class.java)
        startForegroundService(intent)
    }

    private fun stopFlow() {
        AppState.floatingServiceRunning = false
        AppState.screenCaptureServiceRunning = false

        stopService(Intent(this, FloatingService::class.java))
        stopService(Intent(this, ScreenCaptureService::class.java))

        updateButton()
    }

    private fun toast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == OVERLAY_REQUEST_CODE) {
            if (Settings.canDrawOverlays(this)) {
                startFlow()
            } else {
                toast("Overlay permission allow nahi hui")
            }
            return
        }

        if (requestCode == MEDIA_PROJECTION_REQUEST_CODE) {
            if (resultCode == RESULT_OK && data != null) {
                startCaptureService(data)
                startFloatingService()
                updateButton()
            } else {
                toast("Screen capture permission allow nahi hui")
                updateButton()
            }
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        Shizuku.onRequestPermissionResult(requestCode, grantResults)

        if (requestCode == SHIZUKU_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startFlow()
            } else {
                toast("Shizuku permission allow nahi hui")
            }
            return
        }

        if (requestCode == NOTIFICATION_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startFlow()
            } else {
                toast("Notification permission allow nahi hui")
            }
        }
    }
}

class FloatingService : Service() {

    companion object {
        private const val CHANNEL_ID = "cursor_align_floating_channel"
        private const val NOTIFICATION_ID = 2001
    }

    private lateinit var windowManager: WindowManager

    private var iconView: View? = null
    private var menuView: View? = null

    private lateinit var iconParams: WindowManager.LayoutParams
    private lateinit var menuParams: WindowManager.LayoutParams

    override fun onBind(intent: Intent?): IBinder? {
        return null
    }

    override fun onCreate() {
        super.onCreate()

        AppState.floatingServiceRunning = true
        AppState.enabled = Prefs.isEnabled()

        windowManager = getSystemService(WINDOW_SERVICE) as WindowManager

        addIconView()
        addMenuView()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        createNotificationChannel()

        val notification = Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("Cursor Align")
            .setContentText("Floating control active")
            .setSmallIcon(android.R.drawable.ic_menu_info_details)
            .build()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }

        return START_STICKY
    }

    private fun addIconView() {
        val frame = FrameLayout(this)

        val icon = ImageView(this)
        val iconSize = dp(72)
        val iconParams = FrameLayout.LayoutParams(iconSize, iconSize)

        icon.scaleType = ImageView.ScaleType.CENTER_CROP
        loadFloatingIcon(icon)

        icon.setOnClickListener {
            showMenu()
        }

        val close = TextView(this)
        close.text = "X"
        close.setTextColor(Color.WHITE)
        close.gravity = Gravity.CENTER
        close.setBackgroundColor(Color.parseColor("#E53935"))

        val closeSize = dp(28)
        val closeParams = FrameLayout.LayoutParams(closeSize, closeSize, Gravity.TOP or Gravity.END)

        close.setOnClickListener {
            closeApp()
        }

        frame.addView(icon, iconParams)
        frame.addView(close, closeParams)

        iconView = frame

        this.iconParams = WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            getOverlayType(),
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                    WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL or
                    WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
            PixelFormat.TRANSLUCENT
        )

        this.iconParams.gravity = Gravity.TOP or Gravity.START
        this.iconParams.x = 80
        this.iconParams.y = 300

        windowManager.addView(iconView, this.iconParams)
    }

    private fun addMenuView() {
        val menu = LinearLayout(this)
        menu.orientation = LinearLayout.VERTICAL
        menu.s
