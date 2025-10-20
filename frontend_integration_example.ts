/**
 * Frontend Integration Example - React + TypeScript + MediaPipe
 * 
 * Ví dụ code để tích hợp với backend từ React frontend
 */

// Types for MediaPipe landmarks
interface MediaPipeLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

interface MediaPipeResults {
  pose?: MediaPipeLandmark[];      // 25 points upper body only (không phải 33)
  left_hand?: MediaPipeLandmark[]; // 21 points
  right_hand?: MediaPipeLandmark[]; // 21 points
  // face?: REMOVED - không lấy face landmarks nữa
}

interface CaptureFrame {
  timestamp: number;
  landmarks: MediaPipeResults;
}

interface UploadResponse {
  success: boolean;
  task_id: string;
  status: string;
  total_frames: number;
  filename: string;
  detail?: string;
}

// Configuration
const API_BASE_URL = 'http://localhost:8000';
const MAX_UPLOAD_RETRIES = 3;
const UPLOAD_TIMEOUT = 30000; // 30 seconds

/**
 * Service class để upload camera data
 */
class CameraUploadService {
  private frames: CaptureFrame[] = [];
  private sessionId: string = '';
  private startTime: number = 0;

  /**
   * Bắt đầu capture session
   */
  startSession(sessionId?: string): void {
    this.sessionId = sessionId || `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    this.frames = [];
    this.startTime = Date.now();
    console.log(`📹 Started capture session: ${this.sessionId}`);
  }

  /**
   * Thêm frame từ MediaPipe results
   */
  addFrame(results: MediaPipeResults): void {
    if (!this.sessionId) {
      throw new Error('No active session. Call startSession() first.');
    }

    const timestamp = Date.now() - this.startTime;
    
    const frame: CaptureFrame = {
      timestamp,
      landmarks: {
        pose: results.pose || [],
        // face: REMOVED - không lấy face landmarks
        left_hand: results.left_hand || [],
        right_hand: results.right_hand || []
      }
    };

    this.frames.push(frame);
    
    // Optional: Log progress
    if (this.frames.length % 30 === 0) {
      console.log(`📊 Captured ${this.frames.length} frames`);
    }
  }

  /**
   * Upload captured data to backend
   */
  async uploadSession(
    user: string,
    label: string,
    onProgress?: (progress: number) => void
  ): Promise<UploadResponse> {
    if (!this.sessionId || this.frames.length === 0) {
      throw new Error('No data to upload. Capture some frames first.');
    }

    const payload = {
      user,
      label,
      session_id: this.sessionId,
      frames: this.frames
    };

    console.log(`📤 Uploading ${this.frames.length} frames...`);
    
    // Calculate payload size for progress indication
    const payloadSize = JSON.stringify(payload).length;
    console.log(`📦 Payload size: ${(payloadSize / 1024).toFixed(1)} KB`);

    let lastError: Error;

    // Retry logic
    for (let attempt = 1; attempt <= MAX_UPLOAD_RETRIES; attempt++) {
      try {
        if (onProgress) onProgress(0);

        const response = await fetch(`${API_BASE_URL}/upload/camera`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'User-Agent': 'React-MediaPipe-Frontend/1.0'
          },
          body: JSON.stringify(payload),
          signal: AbortSignal.timeout(UPLOAD_TIMEOUT)
        });

        if (onProgress) onProgress(100);

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`HTTP ${response.status}: ${errorText}`);
        }

        const result: UploadResponse = await response.json();
        
        if (!result.success) {
          throw new Error(result.detail || 'Upload failed');
        }

        console.log('✅ Upload successful:', result);
        
        // Clear session data after successful upload
        this.frames = [];
        this.sessionId = '';
        
        return result;

      } catch (error) {
        lastError = error as Error;
        console.warn(`❌ Upload attempt ${attempt} failed:`, error);
        
        if (attempt < MAX_UPLOAD_RETRIES) {
          const delay = Math.pow(2, attempt) * 1000; // Exponential backoff
          console.log(`⏳ Retrying in ${delay}ms...`);
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }

    throw lastError;
  }

  /**
   * Get current session stats
   */
  getSessionStats() {
    return {
      sessionId: this.sessionId,
      frameCount: this.frames.length,
      duration: this.frames.length > 0 ? 
        this.frames[this.frames.length - 1].timestamp : 0,
      estimatedSize: JSON.stringify(this.frames).length
    };
  }

  /**
   * Clear current session
   */
  clearSession(): void {
    this.frames = [];
    this.sessionId = '';
    this.startTime = 0;
  }
}

/**
 * Hook để sử dụng trong React component
 */
function useCameraUpload() {
  const [uploadService] = useState(() => new CameraUploadService());
  const [isCapturing, setIsCapturing] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [stats, setStats] = useState(uploadService.getSessionStats());

  const startCapture = useCallback((sessionId?: string) => {
    uploadService.startSession(sessionId);
    setIsCapturing(true);
    setStats(uploadService.getSessionStats());
  }, [uploadService]);

  const stopCapture = useCallback(() => {
    setIsCapturing(false);
  }, []);

  const addFrame = useCallback((results: MediaPipeResults) => {
    if (isCapturing) {
      uploadService.addFrame(results);
      setStats(uploadService.getSessionStats());
    }
  }, [uploadService, isCapturing]);

  const upload = useCallback(async (user: string, label: string) => {
    if (isUploading) return;

    setIsUploading(true);
    setUploadProgress(0);

    try {
      const result = await uploadService.uploadSession(
        user,
        label,
        setUploadProgress
      );
      
      setStats(uploadService.getSessionStats());
      return result;
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  }, [uploadService, isUploading]);

  const clearSession = useCallback(() => {
    uploadService.clearSession();
    setIsCapturing(false);
    setStats(uploadService.getSessionStats());
  }, [uploadService]);

  return {
    isCapturing,
    isUploading,
    uploadProgress,
    stats,
    startCapture,
    stopCapture,
    addFrame,
    upload,
    clearSession
  };
}

// Generate MediaPipe-like data
function generateFrame(frameIndex: number) {
  return {
    timestamp: frameIndex * 33,
    landmarks: {
      // Chỉ 25 điểm pose upper body + hands
      pose: Array(25).fill(null).map(() => ({x: 0.5, y: 0.5, z: 0.0, visibility: 0.9})),
      // face: REMOVED - không lấy face
      left_hand: Array(21).fill(null).map(() => ({x: 0.3, y: 0.6, z: 0.1})),
      right_hand: Array(21).fill(null).map(() => ({x: 0.7, y: 0.6, z: 0.1}))
    }
  };
}

// Upload to backend
async function uploadFrames(user: string, label: string, frames: CaptureFrame[]): Promise<UploadResponse> {
  const response = await fetch('http://localhost:8000/upload/camera', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      user,
      label,
      session_id: `session_${Date.now()}`,
      frames
    })
  });
  
  return response.json();
}

export { CameraUploadService, useCameraUpload };
export type { MediaPipeResults, CaptureFrame, UploadResponse };