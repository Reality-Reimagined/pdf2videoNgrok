from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import stripe
import os
from dotenv import load_dotenv
import logging
from pdfminer.high_level import extract_text
from deepgram import DeepgramClient, SpeakOptions
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips, ColorClip
import ffmpeg
import assemblyai as aai
import requests
from together import Together
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime
from fastapi.staticfiles import StaticFiles
import time
import random

# Load environment variables
load_dotenv()

# API Keys
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY")
ASSEMBLY_AI_API_KEY = os.getenv("ASSEMBLY_AI_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")

# Configure Stripe
stripe.api_key = STRIPE_SECRET_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["http://localhost:5173"],
    # allow_origins=["*"],
    allow_origins=[
        "http://localhost:5173",
        "https://682d-104-158-73-223.ngrok-free.app"
        "https://loon-stirred-terribly.ngrok-free.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stripe webhook secret
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
app.mount("/static", StaticFiles(directory="."), name="static")

# Your existing endpoints here...
# File Cleanup
def clean_up_files():
    files_to_delete = ["output.wav", "background.mp4", "output_video.mp4", "output_with_subtitles.mp4", "subtitles.srt"]
    for file in files_to_delete:
        if os.path.exists(file):
            os.remove(file)
    logging.info("Temporary files cleaned up.")

# # Extract Text from PDF
# @app.post("/extract-text/")
# async def extract_pdf_text(file: UploadFile = File(...)):
#     try:
#         content = await file.read()
#         with open("temp.pdf", "wb") as f:
#             f.write(content)
#         text = extract_text("temp.pdf")
#         os.remove("temp.pdf")
#         return {"text": text}
#     except Exception as e:
#         logging.error(f"Error extracting text: {e}")
#         raise HTTPException(status_code=500, detail="Error extracting text from PDF.")

@app.post("/extract-text/")
async def extract_pdf_text(file: UploadFile = File(...)):
    try:
        # Debug logging
        logging.info(f"Received file: {file.filename}")
        logging.info(f"Content type: {file.content_type}")
        
        # Validate file type
        if not file.filename.endswith('.pdf'):
            logging.error(f"Invalid file type: {file.filename}")
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF files are allowed. Received: {file.filename}"
            )
            
        # Read content
        content = await file.read()
        
        # Validate content is not empty
        if not content:
            logging.error("Empty file received")
            raise HTTPException(
                status_code=400,
                detail="File is empty"
            )
            
        # Log content size
        logging.info(f"Content size: {len(content)} bytes")
            
        # Save and process file
        with open("temp.pdf", "wb") as f:
            f.write(content)
        
        text = extract_text("temp.pdf")
        
        # Ensure text is a string
        if text is None:
            text = ""
        
        # Always return in the expected format
        return {"text": str(text)}
        
    except Exception as e:
        logging.error(f"Error extracting text: {e}")
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")
        raise HTTPException(status_code=500, detail=str(e))

# Generate Script
class ScriptRequest(BaseModel):
    text: str
    model: str

@app.post("/generate-script/")
async def generate_script(request: ScriptRequest):
    try:
        # Log incoming request
        logging.info(f"Received request with model: {request.model}")
        logging.info(f"Text length: {len(request.text)}")

        if request.model.lower() == "groq" and GROQ_API_KEY:
            logging.info("Using Groq API")
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": request.text}],
                temperature=0.4,
                max_tokens=750,
            )
        elif request.model.lower() == "together" or not GROQ_API_KEY:
            logging.info("Using Together API")
            client = Together(api_key=TOGETHER_API_KEY)
            response = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                messages=[{"role": "user", "content": request.text}],
                temperature=0.4,
                max_tokens=550,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid model selection or missing API keys"
            )

        generated_text = response.choices[0].message.content
        logging.info("Successfully generated script")
        
        return {"script": generated_text}
        
    except Exception as e:
        logging.error(f"Error in generate_script: {str(e)}")
        # Return more specific error message
        raise HTTPException(
            status_code=500,
            detail=f"Error generating script: {str(e)}"
        )

# Text-to-Speech
class TextToSpeechRequest(BaseModel):
    text: str

@app.post("/text-to-speech/")
async def text_to_speech(request: TextToSpeechRequest):
    try:
        
        
        if os.path.exists("output.wav"):
            os.remove("output.wav")  # Remove existing file
            
        deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY)
        options = SpeakOptions(model="aura-zeus-en")
        
        # Use rest.v("1").save instead of synthesize
        response = deepgram.speak.rest.v("1").save(
            "output.wav",
            {"text": request.text},
            options
        )
        
        # Verify file exists and has content
        if not os.path.exists("output.wav"):
            raise HTTPException(status_code=500, detail="Failed to create audio file")
            
        file_size = os.path.getsize("output.wav")
        logging.info(f"Generated audio file size: {file_size} bytes")
        
        return {"message": "Audio generated successfully."}
    except Exception as e:
        logging.error(f"Error generating speech: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Fetch Background Video
async def get_background_video(theme: str) -> str:
    try:
        logging.info(f"Fetching background video for theme: {theme}")
        
        # Construct Pixabay API URL with theme as search query
        base_url = "https://pixabay.com/api/videos/"
        params = {
            "key": PIXABAY_API_KEY,
            "q": theme.strip(),  # Remove any whitespace
            "per_page": 10,
            "orientation": "horizontal",
            "min_duration": 10,
        }
        
        logging.info(f"Searching Pixabay with params: {params}")
        
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        logging.info(f"Found {len(data.get('hits', []))} videos for theme: {theme}")
        
        if not data["hits"]:
            logging.warning(f"No videos found for theme: {theme}, using default theme")
            # Retry with a default theme
            params["q"] = "abstract"
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
        
        # Get random video from results
        video = random.choice(data["hits"])
        video_url = video["videos"]["medium"]["url"]
        
        # Download the video
        response = requests.get(video_url)
        response.raise_for_status()
        
        with open("background.mp4", "wb") as f:
            f.write(response.content)
            
        logging.info(f"Successfully downloaded background video for theme: {theme}")
        return "background.mp4"
        
    except Exception as e:
        logging.error(f"Error getting background video: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting background video: {str(e)}")

# # Create Video
# @app.post("/create-video/")
# async def create_video():
#     try:
#         if os.path.exists("output_video.mp4"):
#             return {"message": "Video already exists."}
#         if not os.path.exists("output.wav") or not os.path.exists("background.mp4"):
#             raise HTTPException(status_code=500, detail="Missing required files.")
#         audio = AudioFileClip("output.wav")
#         video = VideoFileClip("background.mp4")
#         video = video.set_audio(audio).subclip(0, audio.duration)
#         video.write_videofile("output_video.mp4", codec="libx264", audio_codec="aac")
#         return {"message": "Video created successfully."}
#     except Exception as e:
#         logging.error(f"Error creating video: {e}")
#         raise HTTPException(status_code=500, detail="Error creating video.")



@app.post("/create-video/")
async def create_video(theme: str = Form(...)):
    try:
        logging.info(f"Received video generation request with theme: {theme}")
        
        if not os.path.exists("output.wav"):
            raise HTTPException(status_code=400, detail="Audio file not found")
            
        # Get background video based on theme
        background_video = await get_background_video(theme)
        
        if not os.path.exists(background_video):
            raise HTTPException(status_code=400, detail="Background video not found")
        
        # Add debug logging for Pixabay API response
        logging.info(f"Using background video for theme: {theme}")
        
        audio_size = os.path.getsize("output.wav")
        video_size = os.path.getsize(background_video)
        logging.info(f"Audio file size: {audio_size} bytes")
        logging.info(f"Video file size: {video_size} bytes")
        
        try:
            audio = AudioFileClip("output.wav")
            logging.info(f"Audio duration: {audio.duration}s")
        except Exception as e:
            logging.error(f"Error loading audio: {e}")
            raise HTTPException(status_code=500, detail=f"Error loading audio: {str(e)}")
            
        try:
            video = VideoFileClip(background_video)
            logging.info(f"Video duration: {video.duration}s")
        except Exception as e:
            logging.error(f"Error loading video: {e}")
            raise HTTPException(status_code=500, detail=f"Error loading video: {str(e)}")

        # Adjust video duration to match audio
        if video.duration < audio.duration:
            clips = []
            current_duration = 0
            while current_duration < audio.duration:
                clips.append(video)
                current_duration += video.duration
            final_video = concatenate_videoclips(clips).with_duration(audio.duration)
        else:
            final_video = video.with_duration(audio.duration)

        final_video = final_video.with_audio(audio)
        
        try:
            # Use AMD acceleration for final video
            final_video.write_videofile(
                "output_video.mp4",
                codec="h264_amf",
                audio_codec="aac",
                temp_audiofile="temp-audio.m4a",
                remove_temp=True,
                ffmpeg_params=[
                    "-quality", "speed",
                    "-rc", "vbr_peak",
                    "-usage", "ultralowlatency",
                    "-b:v", "2M",
                    "-maxrate", "4M",
                    "-bufsize", "2M",
                    "-profile:v", "high"
                ],
                fps=30,
                threads=4,
                preset="ultrafast"
            )
        except Exception as e:
            logging.error(f"Error writing video: {e}")
            raise HTTPException(status_code=500, detail=f"Error writing video: {str(e)}")

        return {"message": "Video created successfully"}
        
    except Exception as e:
        logging.error(f"Error creating video: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            audio.close()
            video.close()
            final_video.close()
        except:
            pass

# Generate Subtitles
@app.post("/generate-subtitles/")
async def generate_subtitles():
    try:
        # if os.path.exists("subtitles.srt"):
        #     return {"message": "Subtitles already exist."}
        if not ASSEMBLY_AI_API_KEY:
            raise HTTPException(status_code=500, detail="AssemblyAI API key is missing.")
        aai.settings.api_key = ASSEMBLY_AI_API_KEY
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe("output.wav")
        subtitles = transcript.export_subtitles_srt()
        with open("subtitles.srt", "w") as f:
            f.write(subtitles)
        return {"message": "Subtitles generated successfully."}
    except Exception as e:
        logging.error(f"Error generating subtitles: {e}")
        raise HTTPException(status_code=500, detail="Error generating subtitles.")

# # Add Hard Subtitles
# class SubtitleStyleOptions(BaseModel):
#     FontName: str = "Courier"
#     FontSize: int = 42
#     PrimaryColour: str = "&H00FFFFFF"
#     BackColour: str = "&H64000000"
#     BorderStyle: int = 5
#     Outline: int = 8
#     Shadow: int = 1

# @app.post("/add-hard-subtitles/")
# async def add_hard_subtitles():
#     try:
#         input_video = "output_video.mp4"
#         subtitle_file = "subtitles.srt"
#         output_video = "output_with_subtitles.mp4"

#         # Optimized AMD parameters for subtitle addition
#         output_stream = ffmpeg.output(
#             ffmpeg.input(input_video),
#             output_video,
#             vf=f"subtitles='{subtitle_file}':force_style='{style}'",
#             **{
#                 'vcodec': 'h264_amf',
#                 'acodec': 'copy',          # Copy audio instead of re-encoding
#                 'b:v': '2M',
#                 'maxrate': '4M',
#                 'bufsize': '4M',
#                 'quality': 'speed',
#                 'rc': 'vbr_latency',
#                 'usage': 'lowlatency',
#                 'profile:v': 'high',
#                 'level': '4.2',
#                 'bf': '0',
#                 'async_depth': '1',
#                 'frame_skips': '1',
#                 'preanalysis': '0'
#             }
#         )
        
#         start_time = time.time()
#         ffmpeg.run(output_stream, overwrite_output=True, cmd='ffmpeg', capture_stderr=True)
#         duration = time.time() - start_time
        
#         # Get actual file sizes
#         input_size = os.path.getsize(input_video)
#         output_size = os.path.getsize(output_video)
        
#         logging.info(f"""
# Video Processing Complete:
# - Processing Time: {duration:.2f} seconds
# - Input Size: {input_size/1024/1024:.2f} MB (Raw bytes: {input_size})
# - Output Size: {output_size/1024/1024:.2f} MB (Raw bytes: {output_size})
# - Compression Ratio: {input_size/output_size:.2f}x
# - Bitrate: {(output_size*8)/(duration*1024*1024):.2f} Mbps
# """)
        
#         return {"message": "Subtitles added successfully"}
        
#     except ffmpeg.Error as e:
#         error_message = e.stderr.decode() if hasattr(e, 'stderr') else str(e)
#         logging.error(f"FFmpeg error: {error_message}")
#         raise HTTPException(status_code=500, detail=f"FFmpeg error: {error_message}")
        
#     except Exception as e:
#         logging.error(f"Error adding hard subtitles: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


@app.post("/add-hard-subtitles/")
async def add_hard_subtitles():
    try:
        input_video = "output_video.mp4"
        subtitle_file = "subtitles.srt"
        output_video = "output_with_subtitles.mp4"

        # Define subtitle style
        style = "FontName=Arial,FontSize=42,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,BorderStyle=3"

        # Check GPU support first
        try:
            gpu_info = ffmpeg.probe('color=c=black:s=1280x720:r=30', f='lavfi')
            logging.info(f"GPU Info: {gpu_info}")
        except Exception as e:
            logging.warning(f"Error checking GPU support: {e}")

        # Add detailed logging for GPU detection
        logging.info("Attempting AMD hardware encoding...")
        
        try:
            # Create input stream
            video_input_stream = ffmpeg.input(input_video)
            
            # Configure output with AMD optimizations
            output_stream = ffmpeg.output(
                video_input_stream,
                output_video,
                vf=f"subtitles='{subtitle_file}':force_style='{style}'",
                **{
                    'c:v': 'h264_amf',         # Explicit codec name
                    'c:a': 'copy',             # Copy audio
                    'b:v': '2M',
                    'maxrate': '4M',
                    'bufsize': '2M',
                    'quality': 'speed',
                    'rc': 'vbr_peak',
                    'usage': 'ultralowlatency',
                    'profile:v': 'high'
                }
            )
            
            # Add debug logging
            cmd = ffmpeg.compile(output_stream)
            logging.info(f"FFmpeg command: {' '.join(cmd)}")
            
            start_time = time.time()
            ffmpeg.run(output_stream, overwrite_output=True)
            duration = time.time() - start_time
            
            # Get file sizes
            input_size = os.path.getsize(input_video)
            output_size = os.path.getsize(output_video)
            
            logging.info(f"""
Video Processing Complete:
- Processing Time: {duration:.2f} seconds
- Input Size: {input_size/1024/1024:.2f} MB (Raw bytes: {input_size})
- Output Size: {output_size/1024/1024:.2f} MB (Raw bytes: {output_size})
- Compression Ratio: {input_size/output_size:.2f}x
- Bitrate: {(output_size*8)/(duration*1024*1024):.2f} Mbps
""")
            
            return {"message": "Subtitles added successfully"}
            
        except ffmpeg.Error as e:
            error_message = e.stderr.decode() if hasattr(e, 'stderr') else str(e)
            logging.error(f"FFmpeg error: {error_message}")
            raise HTTPException(status_code=500, detail=f"FFmpeg error: {error_message}")
            
    except Exception as e:
        logging.error(f"Error adding hard subtitles: {e}")
        raise HTTPException(status_code=500, detail=str(e))
# Cleanup Endpoint
@app.post("/clean-up/")
async def clean_up():
    clean_up_files()
    return {"message": "Temporary files cleaned up."}


@app.post("/create-checkout-session")
async def create_checkout_session(request: dict):
    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': request.get('priceId'),
                'quantity': 1,
            }],
            mode='subscription',
            success_url='http://localhost:5173/dashboard?session_id={CHECKOUT_SESSION_ID}',
            cancel_url='http://localhost:5173/dashboard',
        )
        return JSONResponse(content={"id": checkout_session.id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/webhook")
# async def stripe_webhook(request: Request):
#     try:
#         # Get the webhook secret from environment variable
#         webhook_secret = STRIPE_WEBHOOK_SECRET
        
#         # Get the webhook data
#         payload = await request.body()
#         sig_header = request.headers.get('stripe-signature')
        
#         try:
#             event = stripe.Webhook.construct_event(
#                 payload, sig_header, webhook_secret
#             )
#         except ValueError as e:
#             raise HTTPException(status_code=400, detail='Invalid payload')
#         except stripe.error.SignatureVerificationError as e:
#             raise HTTPException(status_code=400, detail='Invalid signature')
        
#         # Handle the event
#         if event['type'] == 'customer.subscription.updated':
#             subscription = event['data']['object']
#             # Update user's subscription status in your database
#             # This is where you'd update the user's subscription status
#             logging.info(f"Subscription updated: {subscription.id}")
            
#         elif event['type'] == 'customer.subscription.deleted':
#             subscription = event['data']['object']
#             # Handle subscription cancellation
#             logging.info(f"Subscription cancelled: {subscription.id}")
            
#         return JSONResponse(content={"status": "success"})
#     except Exception as e:
#         logging.error(f"Error processing webhook: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

class Video(BaseModel):
    title: str
    description: str
    videoPath: str
    createdAt: str
    hasSubtitles: bool

# In-memory storage (replace with database in production)
videos_db: List[Video] = []

@app.get("/videos/")
async def get_videos():
    return videos_db

@app.post("/videos/")
async def log_video(video: Video):
    videos_db.append(video.dict())
    return {"message": "Video logged successfully"}

# # Add this to serve video files
# from fastapi.staticfiles import StaticFiles
# app.mount("/videos", StaticFiles(directory="."), name="videos")

# app.include_router(stripe_router, prefix="/api/stripe", tags=["stripe"])

@app.get("/api/stripe/verify-session")
async def verify_session(session_id: str):
    try:
        # Retrieve the session from Stripe
        session = stripe.checkout.Session.retrieve(session_id)
        
        # Get the subscription details
        subscription = stripe.Subscription.retrieve(session.subscription)
        
        # Determine the plan type based on the price ID
        plan_type = 'pro' if subscription.plan.id == STRIPE_PRO_PRICE_ID else 'enterprise'
        
        return {
            "subscription": {
                "id": subscription.id,
                "plan": plan_type,
                "current_period_end": subscription.current_period_end,
                "status": subscription.status
            }
        }
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))



# Add this new endpoint
@app.get("/videos/{filename}")
async def get_video(filename: str):
    try:
        video_path = filename
        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Add debug logging
        file_size = os.path.getsize(video_path)
        logging.info(f"Serving video file: {filename}, size: {file_size} bytes")
        
        return FileResponse(
            path=video_path,
            media_type="video/mp4",
            filename=filename,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Type": "video/mp4"
            }
        )
    except Exception as e:
        logging.error(f"Error serving video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def check_gpu_support():
    try:
        # Check if AMD GPU encoding is available
        result = ffmpeg.probe(
            None,
            v='error',
            select_streams='v:0',
            show_entries='codec_name',
            _options=['-encoders']
        )
        encoders = result.get('encoders', '')
        has_amd = 'h264_amf' in encoders
        logging.info(f"AMD GPU encoding {'available' if has_amd else 'not available'}")
        return has_amd
    except Exception as e:
        logging.warning(f"Error checking GPU support: {e}")
        return False

# Call this when your app starts
HAS_GPU_SUPPORT = check_gpu_support()

# Add this function to check FFmpeg AMD support
def verify_amd_support():
    try:
        import subprocess
        
        # Check FFmpeg version and encoders
        result = subprocess.run(['ffmpeg', '-encoders'], 
                              capture_output=True, 
                              text=True)
        
        # Look for AMD encoders
        encoders = result.stdout
        amd_encoders = [
            'h264_amf',
            'hevc_amf',
            'av1_amf'
        ]
        
        available_encoders = []
        for encoder in amd_encoders:
            if encoder in encoders:
                available_encoders.append(encoder)
        
        if available_encoders:
            logging.info(f"AMD GPU encoders available: {', '.join(available_encoders)}")
            return True
        else:
            logging.warning("No AMD GPU encoders found")
            return False
            
    except Exception as e:
        logging.error(f"Error checking AMD support: {e}")
        return False



def monitor_gpu_usage():
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            logging.info(f"""
GPU Status:
- Name: {gpu.name}
- Load: {gpu.load*100}%
- Memory Use: {gpu.memoryUsed}MB/{gpu.memoryTotal}MB
- Temperature: {gpu.temperature}°C
""")
    except ImportError:
        logging.warning("GPUtil not installed. Skip GPU monitoring.")
        pass

# Install GPUtil
# pip install gputil

# Add this to your startup code
if __name__ == "__main__":
    has_amd = verify_amd_support()
    if has_amd:
        logging.info("AMD GPU acceleration is ready")
    else:
        logging.warning("Using CPU fallback for video processing")

def test_amd_encoding():
    try:
        # Test file paths
        input_video = "output_video.mp4"  # Use an existing video file
        test_outputs = {
            'h264': 'test_h264_gpu.mp4',
            'hevc': 'test_hevc_gpu.mp4',
            'av1': 'test_av1_gpu.mp4'
        }

        for encoder, output_file in test_outputs.items():
            start_time = time.time()
            
            # Create input stream
            stream = ffmpeg.input(input_video)
            
            # Configure output with AMD encoder
            output_stream = ffmpeg.output(
                stream,
                output_file,
                vcodec=f'{encoder}_amf',
                acodec='aac',
                # AMD optimizations
                rc='vbr_peak',
                quality='speed',
                usage='ultralowlatency'
            )
            
            # Run encoding
            logging.info(f"Testing {encoder}_amf encoder...")
            ffmpeg.run(output_stream, overwrite_output=True)
            
            # Calculate processing time
            duration = time.time() - start_time
            
            # Get file sizes
            input_size = os.path.getsize(input_video)
            output_size = os.path.getsize(output_file)
            
            logging.info(f"""
{encoder.upper()} Test Results:
- Processing Time: {duration:.2f} seconds
- Input Size: {input_size/1024/1024:.2f} MB
- Output Size: {output_size/1024/1024:.2f} MB
- Compression Ratio: {input_size/output_size:.2f}x
""")
            
            # Clean up test file
            os.remove(output_file)
            
        return True
        
    except ffmpeg.Error as e:
        logging.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
        return False
    except Exception as e:
        logging.error(f"Test error: {str(e)}")
        return False

# Add this endpoint to run the test
@app.post("/test-gpu/")
async def test_gpu():
    try:
        result = test_amd_encoding()
        if result:
            return {"message": "GPU encoding test successful! Check server logs for details."}
        else:
            raise HTTPException(status_code=500, detail="GPU encoding test failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def create_video_with_audio(audio_file, duration, output_file="output_video.mp4"):
    try:
        # Log AMD capabilities
        logging.info("Checking AMD encoder availability...")
        
        input_audio = ffmpeg.input(audio_file)
        video = ffmpeg.input(
            'color=c=black:s=1280x720:r=30',
            f='lavfi',
            t=duration
        )
        
        # Optimized AMD parameters
        output = ffmpeg.output(
            video,
            input_audio,
            output_file,
            **{
                'c:v': 'h264_amf',         # Explicit codec name
                'c:a': 'aac',
                'b:v': '2M',
                'maxrate': '4M',
                'bufsize': '2M',
                'quality': 'speed',
                'rc': 'vbr_peak',
                'usage': 'ultralowlatency',
                'profile:v': 'high'
            }
        )
        
        # Log the actual ffmpeg command
        cmd = ffmpeg.compile(output)
        logging.info(f"FFmpeg command: {' '.join(cmd)}")
        
        start_time = time.time()
        ffmpeg.run(output, overwrite_output=True)
        process_time = time.time() - start_time
        
        # Log performance metrics
        output_size = os.path.getsize(output_file)
        logging.info(f"""
Video Creation Stats:
- Processing Time: {process_time:.2f} seconds
- Output Size: {output_size/1024/1024:.2f} MB
- Processing Speed: {(output_size/1024/1024)/process_time:.2f} MB/s
""")
        
        return True
        
    except Exception as e:
        logging.error(f"Error creating video: {str(e)}")
        return False