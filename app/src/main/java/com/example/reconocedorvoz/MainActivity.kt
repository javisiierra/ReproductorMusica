package com.example.reconocedorvoz

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.media.AudioManager
import android.media.MediaPlayer
import android.os.Bundle
import android.util.Log
import android.view.MotionEvent
import android.widget.ImageButton
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.support.audio.TensorAudio
import org.tensorflow.lite.task.audio.classifier.AudioClassifier
import org.tensorflow.lite.task.audio.classifier.AudioClassifier.AudioClassifierOptions
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    // Componentes de audio
    private var mediaPlayer: MediaPlayer? = null
    private var audioManager: AudioManager? = null
    private var audioClassifier: AudioClassifier? = null
    private var tensorAudio: TensorAudio? = null

    // Control de ejecución
    private var isListening = false
    private val executor = Executors.newSingleThreadExecutor()
    private var lastCommandTime = 0L
    private var lastExecutedCommand: String? = null
    private val COMMAND_COOLDOWN = 1500L
    private val COMMAND_TIMEOUT = 2000L
    private val THRESHOLD = 0.75f

    // Datos de canciones
    private var currentSongIndex = 0
    private val songs = arrayOf(R.raw.cancion1, R.raw.cancion2, R.raw.cancion3)
    private var isPrepared = false

    // Configuración del modelo
    private val MODEL_PATH = "modelo_comandos.tflite"
    private val CLASS_LABELS = listOf(
        "0 Anterior Canción",
        "1 Apagar",
        "2 Bajar volumen",
        "3 Encender",
        "4 Ruido de fondo",
        "5 Siguiente canción",
        "6 Subir volumen"
    )

    // Vistas
    private lateinit var tvSongTitle: TextView
    private lateinit var tvArtist: TextView
    private lateinit var recognizedText: TextView
    private lateinit var btnPrevious: ImageButton
    private lateinit var btnPlayPause: ImageButton
    private lateinit var btnNext: ImageButton
    private lateinit var btnMic: ImageButton

    // Permisos
    private val REQUIRED_PERMISSION = Manifest.permission.RECORD_AUDIO
    private val PERMISSION_REQUEST_CODE = 1001

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        initViews()
        setupButtonListeners()
        checkAndRequestPermissions()
    }

    private fun initViews() {
        tvSongTitle = findViewById(R.id.tvSongTitle)
        tvArtist = findViewById(R.id.tvArtist)
        recognizedText = findViewById(R.id.recognizedText)
        audioManager = getSystemService(AUDIO_SERVICE) as AudioManager
        btnPrevious = findViewById(R.id.btnPrevious)
        btnPlayPause = findViewById(R.id.btnPlayPause)
        btnNext = findViewById(R.id.btnNext)
        btnMic = findViewById(R.id.btnMic)
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun setupButtonListeners() {
        btnPlayPause.setOnClickListener { togglePlayPause() }
        btnNext.setOnClickListener { executeCommand("5 Siguiente canción") }
        btnPrevious.setOnClickListener { executeCommand("0 Anterior Canción") }

        btnMic.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    if (!isListening) {
                        startVoiceRecognition()
                        showToast("🎤 Escuchando...")
                    }
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    if (isListening) {
                        stopVoiceRecognition()
                        showToast("🛑 Reconocimiento detenido")
                    }
                    true
                }
                else -> false
            }
        }
    }

    private fun togglePlayPause() {
        if (mediaPlayer?.isPlaying == true) executeCommand("1 Apagar") else executeCommand("3 Encender")
    }

    private fun checkAndRequestPermissions() {
        when {
            ContextCompat.checkSelfPermission(this, REQUIRED_PERMISSION) == PackageManager.PERMISSION_GRANTED -> {
                initializeVoiceRecognition()
                showToast("🎤 Reconocimiento de voz desactivado")
            }
            ActivityCompat.shouldShowRequestPermissionRationale(this, REQUIRED_PERMISSION) -> {
                showToast("Permiso de micrófono requerido")
                requestPermissions(arrayOf(REQUIRED_PERMISSION), PERMISSION_REQUEST_CODE)
            }
            else -> requestPermissions(arrayOf(REQUIRED_PERMISSION), PERMISSION_REQUEST_CODE)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQUEST_CODE && grantResults.isNotEmpty()) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                initializeVoiceRecognition()
                showToast("🎤 Reconocimiento de voz desactivado")
            } else {
                showToast("Funcionalidad de voz desactivada")
            }
        }
    }

    private fun initializeVoiceRecognition() {
        try {
            setupAudioClassifier()
        } catch (e: Exception) {
            showToast("Error en reconocimiento de voz: ${e.message}")
            Log.e("VoiceRecognition", "Error inicializando", e)
        }
    }

    private fun setupAudioClassifier() {
        try {
            val options = AudioClassifierOptions.builder()
                .setScoreThreshold(THRESHOLD)
                .setMaxResults(1)
                .build()

            audioClassifier = AudioClassifier.createFromFileAndOptions(this, MODEL_PATH, options)
            tensorAudio = audioClassifier?.createInputTensorAudio()

            val modelLabels = loadModelLabels()
            if (modelLabels != CLASS_LABELS) {
                Log.w("ModelCheck", "Etiquetas del modelo: $modelLabels\nEsperadas: $CLASS_LABELS")
            }
        } catch (e: Exception) {
            throw Exception("Error cargando modelo: ${e.message}")
        }
    }

    private fun loadModelLabels(): List<String> {
        return try {
            assets.open("labels.txt").bufferedReader().useLines { lines ->
                lines.toList()
            }
        } catch (e: Exception) {
            Log.e("ModelCheck", "Error cargando labels.txt: ${e.message}")
            emptyList()
        }
    }

    private fun initializeMediaPlayer(shouldPlay: Boolean = false) {
        mediaPlayer?.release()
        mediaPlayer = MediaPlayer().apply {
            setDataSource(resources.openRawResourceFd(songs[currentSongIndex]))
            setOnPreparedListener {
                isPrepared = true
                updatePlaybackState()
                updateSongInfo()
                if (shouldPlay) {
                    start()
                    showToast("▶ Reproduciendo")
                }
            }
            setOnCompletionListener {
                playNextSong()
            }
            prepareAsync()
        }
    }

    private fun startVoiceRecognition() {
        if (audioClassifier == null) return

        val audioRecord = audioClassifier!!.createAudioRecord().apply {
            startRecording()
        }

        isListening = true
        lastExecutedCommand = null
        lastCommandTime = 0L

        executor.execute {
            while (isListening) {
                try {
                    tensorAudio?.load(audioRecord)
                    val results = audioClassifier!!.classify(tensorAudio)

                    results.firstOrNull()?.categories?.firstOrNull { category ->
                        category.score > THRESHOLD &&
                                category.label in CLASS_LABELS &&
                                category.label != "4 Ruido de fondo"
                    }?.let { category ->
                        val currentTime = System.currentTimeMillis()
                        val shouldExecute = category.label != lastExecutedCommand ||
                                (currentTime - lastCommandTime) > COMMAND_TIMEOUT

                        if (shouldExecute && (currentTime - lastCommandTime) > COMMAND_COOLDOWN) {
                            lastCommandTime = currentTime
                            lastExecutedCommand = category.label
                            handleVoiceCommand(category.label, category.score)
                            Thread.sleep(COMMAND_COOLDOWN)
                        }
                    }
                } catch (e: Exception) {
                    Log.e("VoiceRecognition", "Error: ${e.message}")
                    runOnUiThread { recognizedText.text = "Error procesando audio" }
                    Thread.sleep(500)
                }
            }
            audioRecord.stop()
            audioRecord.release()
        }
    }

    private fun stopVoiceRecognition() {
        isListening = false
        lastExecutedCommand = null
        lastCommandTime = 0L

        runOnUiThread {
            recognizedText.text = "Esperando comando..."
        }
    }

    private fun handleVoiceCommand(command: String, confidence: Float) {
        runOnUiThread {
            recognizedText.text = "$command (${"%.0f%%".format(confidence * 100)})"
            executeCommand(command)
        }
    }

    private fun executeCommand(command: String) {
        when (command) {
            "3 Encender" -> handlePlay()
            "1 Apagar" -> handlePause()
            "5 Siguiente canción" -> playNextSong()
            "0 Anterior Canción" -> playPreviousSong()
            "6 Subir volumen" -> adjustVolumeUp()
            "2 Bajar volumen" -> adjustVolumeDown()
            "4 Ruido de fondo" -> showToast("Comando no reconocido")
        }
    }

    private fun handlePlay() {
        if (mediaPlayer == null) {
            initializeMediaPlayer(true)
        } else if (!isPrepared) {
            showToast("⏳ Esperando preparación...")
        } else if (mediaPlayer?.isPlaying == false) {
            mediaPlayer?.start()
            updatePlaybackState()
            showToast("▶ Reproduciendo")
        }
    }

    private fun handlePause() {
        mediaPlayer?.pause()
        updatePlaybackState()
        showToast("⏸ Pausado")
    }

    private fun adjustVolumeUp() {
        audioManager?.adjustVolume(AudioManager.ADJUST_RAISE, AudioManager.FLAG_PLAY_SOUND)
        showToast("🔊 Volumen +")
    }

    private fun adjustVolumeDown() {
        audioManager?.adjustVolume(AudioManager.ADJUST_LOWER, AudioManager.FLAG_PLAY_SOUND)
        showToast("🔉 Volumen -")
    }

    private fun updatePlaybackState() {
        val isPlaying = mediaPlayer?.isPlaying ?: false
        btnPlayPause.setImageResource(
            if (isPlaying) R.drawable.ic_pause else R.drawable.ic_play
        )
    }

    private fun playNextSong() {
        currentSongIndex = (currentSongIndex + 1) % songs.size
        initializeMediaPlayer(true)
    }

    private fun playPreviousSong() {
        currentSongIndex = if (currentSongIndex - 1 < 0) songs.size - 1 else currentSongIndex - 1
        initializeMediaPlayer(true)
    }

    private fun updateSongInfo() {
        tvSongTitle.text = "Canción ${currentSongIndex + 1}"
        tvArtist.text = "Artista"
    }

    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }

    override fun onPause() {
        super.onPause()
        isListening = false
        mediaPlayer?.pause()
    }

    override fun onResume() {
        super.onResume()
        // Reconocimiento de voz solo se activa al presionar el botón
        // NO iniciar automáticamente aquí
    }

    override fun onDestroy() {
        super.onDestroy()
        isListening = false
        executor.shutdownNow()
        mediaPlayer?.release()
        mediaPlayer = null
    }
}