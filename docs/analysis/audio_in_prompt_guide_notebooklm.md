To successfully generate dynamic video from a static image while driving the scene with a frozen audio file (noise=0), you must carefully balance your prompt to bridge the gap between the visual starting point and the acoustic target. LTX-2 is highly sensitive to how these modalities are described and can easily "break" if the prompt, image, and audio provide conflicting instructions.

Here is a detailed guide on best practices, considerations, and examples for this specific Image-to-Video (I2V) and Audio-to-Video workflow.

### 1. Describing the Input Image (The Visual Anchor)
**You must describe the input image perfectly.** If your prompt describes a character, outfit, or environment that is not in your starting image, LTX-2 will often ignore the reference image entirely and switch to generating a pure Text-to-Video (T2V) result, or it will simply freeze the frame.
*   **The "LLM" Trick:** Because LTX-2 prefers extreme detail, many users achieve the best results by passing their input image through a Vision-Language Model (like Qwen-VL or GPT-4o) to generate a highly accurate, literal description of the scene, which is then used as the base of the prompt.
*   **Avoid Overloading:** If you try to prompt for actions or elements that require a completely different camera angle or setting than your input image, the model will struggle. Start with what is currently visible, then describe the transition.

### 2. Prompting for Audio and Lip-Sync
Even though the model is receiving the actual audio file to drive the generation, you must still explicitly tell the model what is happening acoustically.
*   **Transcribe the Dialogue:** Always include the exact spoken words in quotation marks. For example, write: `The man speaks in a harsh low voice and says: "Your exact transcript here."`. While some users have reported occasional success by simply writing "a person is talking" and letting the audio file do the work, developers and experienced workflow builders strongly recommend including the transcript so the model knows precisely where to pad the lip-sync.
*   **Describe the Delivery:** The visual performance needs to match the audio's tone. Describe the character's cadence, accent, and emotional delivery (e.g., "in a sultry begging voice," or "brisk delivery, English accent"). 
*   **The Volume Trick:** If the character in your generation refuses to move their mouth and the video looks like a documentary with a voice-over, try increasing the volume of your input audio file. Louder, peaking audio forces the model to drive the lip-sync harder.
*   **Prompting the Action First:** Structuring your sentence so the physical action precedes the dialogue often yields better results. Use formats like: `The person speaks in a harsh low voice and says "..."` rather than putting the dialogue first.

### 3. Structuring the Prompt
LTX-2 requires long, descriptive, chronological prompts—ideally structured as a single flowing paragraph of up to 200 words. 
*   **Shot Scale & Camera:** Start by establishing the shot type (e.g., "tight cinematic close-up", "static medium shot"). Match your details to the scale; close-ups need heavy facial descriptions, while wide shots need more environmental detail. 
*   **Present Tense Action:** Write the core action as a natural sequence flowing from beginning to end using present tense verbs. 
*   **Avoid Internal Emotions:** Do not tell the model a character is "sad" or "confused." Instead, describe the physical cues: "furrowed brow," "tremor of the chin," or "tears welling".

### 4. Technical Considerations & Workarounds
*   **Fixing "Frozen" Videos:** A notorious issue with I2V generation in LTX-2 is that the video remains completely static. To fix this, you must add subtle h.264 video compression artifacts to your input image using the `LTXVPreprocess` node (often at a strength of 33 to 40). Because LTX-2 was trained on compressed videos, it interprets a pristine, artifact-free image as a static photograph. Adding compression noise tricks the model into recognizing it as a video frame, forcing it to generate motion.
*   **Preventing Likeness Loss (Over-Emoting):** LTX-2 has a tendency to make characters overly expressive during audio-driven generations, which can heavily distort their facial structure and ruin the likeness of your input image. To combat this, use negative prompts like `exaggerated expressions, warped facial features, identity drift`. You can also slightly lower the audio-to-video attention scale in your workflow to calm the facial movements down.

### Concrete Examples

**Example 1: Dialogue from a Static Image**
> *"A tight cinematic close-up of a male doctor speaking directly to the camera inside a modern health consultory. He wears a crisp white lab coat over a light blue shirt, subtle stubble, calm confident expression. Soft diffused daylight enters from a side window, creating gentle highlights on his face and clean shadows. The background is softly blurred with medical shelves and diagnostic equipment. The camera is locked in a shallow-depth close-up using a 50mm lens, with a very subtle push-in as he speaks, maintaining eye contact. Natural skin texture, realistic pores, professional medical atmosphere. Quiet room tone ambience, no music. He says: 'We need to run the tests again immediately, the results are inconclusive.'"*

**Example 2: High-Energy Music Video (Lip-Sync and Rhythm)**
> *"Superman and Lois Lane perform together in a gritty rap music video. Their recognizable appearance and facial identity must remain consistent throughout the scene. At the beginning Lois Lane reacts to the beat with playful rhythmic hype sounds while looking at Superman. The video alternates naturally between different music video shot types: wide shots showing both performers interacting with confident body language, and medium performance shots. Superman begins rapping with intense rhythmic delivery, strong mouth articulation and expressive lip movements while alternating his gaze between Lois Lane and the camera. He performs with sharp rap gestures and confident stage presence. When his line ends, Lois Lane steps forward and answers with her verse. Strong rap performance, alternating voices, expressive mouth movement, clear lip sync, preserve character likeness."*

**Example 3: Stylized / Non-Verbal Audio Reactivity**
> *"Live Action Mode futuristic fashion-dance tableau, neon sci-fi editorial: a dark-skinned dancer with a large textured afro is frozen in a dramatic off-balance tilt, wearing a reflective chrome set. Background is a luminous rectangular LED frame with blue/magenta rim lighting. At the start she holds the extreme lean pose, then slowly wakes into motion—micro tremor in shoulders, fingertips flex. Halfway through she transitions into a smooth hinge and recovery, moving rhythmically to the beat. Toward the end she rotates her head toward camera, eyes lock, and she breathes out one line: 'Watch me bend the light.' The camera makes a controlled slow push-in. Audio: low neon room hum, soft breath, faint fabric creak, subtle whoosh synced to arm sweep, minimal futuristic pulse very low in the mix."*
