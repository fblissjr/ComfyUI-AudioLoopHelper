When passing an audio file into LTX-2 to drive lip sync, your text prompt remains a critical tool for guiding the character's acting, expressions, and timing. Even with audio provided, the model relies on the text to understand the context of the facial movements. 

Here are the best practices for writing lip sync prompts based on community testing and developer advice:

**1. Transcribe the Exact Dialogue**
While some users have reported occasional success by simply writing "a person is talking", the consensus—including direct advice from LTX developers—is that you should explicitly transcribe the spoken words inside quotation marks. 
*   **Format:** `The man says: "Your exact audio transcript goes here."` 
*   If you are extending an existing video or inpainting, you must include the transcript for the *entire* video's audio, including the parts you are leaving unedited, so the model understands the full context.

**2. Describe the Vocal Delivery and Emotion**
The prompt must match the tone, pitch, and accent of your audio file. If you provide a high-pitched audio file but your prompt says "deep voice," the lip sync will likely fail to synchronize. 
*   Describe the character's emotional cadence, such as "in a sultry begging voice," "speaks with great passion," or "speaking in a thick Australian accent".
*   **The "Shouting" Trick:** If the generated lip movement is too subtle or stiff, changing your prompt from "speaking" to "shouting" will force the model to exaggerate the mouth movements.

**3. Use Tight Framing**
Lip sync generations work significantly better in tight close-up or chest-up medium shots. If the character's face is too small or distant in the frame, the model struggles to maintain their identity and lacks the resolution to animate the lips accurately. 

**4. Highlight the Action**
It helps to explicitly tell the model to focus on the mouth. Using phrases like "expressive mouth movement, clear lip sync" or "we clearly see her lips moving in time with the speaking" reinforces the audio-to-video alignment.

**Example of an optimized lip sync prompt:**
> *"A tight cinematic close-up of a blonde woman. The camera is locked off and static. She looks directly at the camera with a serious, determined expression. She speaks with great passion in a harsh, low voice, her lips moving with every spoken word. She says: 'I'm going to make you talk if it's the last thing I do.' Room tone ambience, no music."*

**Troubleshooting Tip:** If your character still looks like they are in a documentary with a voiceover rather than actually speaking the lines, try increasing the volume of your input audio file, as louder audio data often forces the model to drive the lip sync harder.

---

Interestingly, there is actually some debate in the ComfyUI community about whether transcribing the lyrics is strictly required when you provide an audio file! 

Some users have found that writing out full lyrics is "totally unnecessary" and that the "audio alone will do". Because LTX-2 processes the actual mel spectrogram of the audio, the intensity, rhythm, and modulation of the vocal track are often enough to drive the lip sync on their own. Users have reported success by simply prompting that "a man is talking to a woman" or "a woman is singing" and letting the audio file do the rest.

However, experienced workflow builders still recommend transcribing the exact speech or lyrics as a **best practice**. Here is why you might still want to do it:

*   **Precision and Padding:** LTX-2 uses a joint architecture where the text, video, and audio all influence each other. Including the exact text helps the model know exactly "where to pad with precision". It reinforces the audio data with semantic meaning, giving the model a much clearer map of what mouth shapes to form and when.
*   **Preventing Gibberish:** When dealing with heavy background music, complex pronunciations, or fast singing, the audio track alone might confuse the model. Some users have reported that if they leave the dialogue out of the prompt, the generated lip movements devolve into "gibberish". 

**The takeaway:** You don't *have* to transcribe the lyrics every time. You can absolutely try a simple prompt first (e.g., "she sings into the microphone") to save time. But if the character's lip sync looks sloppy or fails to match the beat, adding the exact lyrics back into the prompt is the best way to lock the performance in.

---

It can seem redundant since the model already has the full audio data, but there is actually an active debate in the community about this exact point. 

Here is what users and developers have discovered about whether you truly need to transcribe the audio:

**Why you often DON'T need it:**
Because LTX-2 processes the audio directly, it naturally recognizes the sounds of speech. Many users have found that writing out the exact dialogue or lyrics is completely unnecessary. They report getting perfect lip sync by simply describing the scene—such as "a man is talking to a woman"—and letting the provided audio track do all the heavy lifting to drive the mouth movements. The model inherently "knows what talking is" just by listening to the file.

**Why it is still recommended as a "best practice":**
Despite the successes above, other users have reported that if they leave the spoken text out of the prompt, the character's mouth movements devolve into visual "gibberish". 

Because of this inconsistency, LTX developers and experienced community members still recommend including the exact dialogue as a best practice. Here is why:
*   **Precision:** Giving the model the exact text provides extra semantic context, which helps it know exactly "where to pad with precision" to match the mouth shapes to the audio. 
*   **Complex Scenes:** If your audio has background noise, or if you are generating a scene where there is a "crowd of people talking over each other," the audio alone might confuse the model. In these cases, LTX developers advise using elaborate prompts (e.g., "Person A says: XXXXX, Person B says: YYYYYYY") to force the model to assign the right lip sync to the right character.

**The practical takeaway:** You can absolutely save time by leaving the exact dialogue out of your prompt and just describing the character "talking" or "singing." However, if the resulting lip sync looks sloppy, unconvincing, or turns into gibberish, adding the exact transcript back into the prompt is the best way to lock in the accuracy.

---

Here are several excellent examples of prompts used specifically for generating music videos and lip-syncing to songs, drawn from official guides and community testing. Notice how they carefully describe the vocal delivery, camera movements, rhythmic actions, and specific mouth articulations:

**1. The Intimate Acoustic Performance**
This prompt is great for slower, moodier songs where you want the camera to focus on a grounded, emotional performance:
> "A warm, intimate cinematic performance inside a cozy, wood-paneled bar, lit with soft amber practical lights and shallow depth of field that creates glowing bokeh in the background. The shot opens in a medium close-up on a young female singer in her 20s with short brown hair and bangs, singing into a microphone while strumming an acoustic guitar, her eyes closed and posture relaxed. The camera slowly arcs left around her, keeping her face and mic in sharp focus as two male band members playing guitars remain softly blurred behind her. Warm light wraps around her face and hair as framed photos and wooden walls drift past in the background. Ambient live music fills the space, led by her clear vocals over gentle acoustic strumming." 

**2. Multi-Character Rap / Dynamic Exchanges**
When you have multiple voices in a song (like a duet or rap battle), you need to clearly direct who is speaking and how they react to each other:
> "Superman and Lois Lane perform together in a gritty rap music video. Their recognizable appearance and facial identity must remain consistent throughout the scene. At the beginning Lois Lane reacts to the beat with playful rhythmic hype sounds while looking at Superman, then briefly glancing at the camera with a teasing confident smile. The video alternates naturally between different music video shot types: wide shots showing both performers interacting with confident body language, medium performance shots capturing their rap delivery and movement, and occasional close-up reaction shots highlighting facial expressions and lip sync. Superman begins rapping with intense rhythmic delivery, strong mouth articulation and expressive lip movements while alternating his gaze between Lois Lane and the camera. He performs with sharp rap gestures and confident stage presence. When his line ends, Lois Lane steps forward and answers with her verse, rapping with energetic delivery and expressive lip movements while Superman reacts with amused approval."

**3. The Solo "Stage" Performance (Stylized)**
This is highly effective for keeping the model focused purely on the singer's rhythmic body mechanics and lip-sync without background distractions:
> "A single, completely solitary humanoid Shiba Inu performer sings passionately into a handheld microphone. No other people, animals, silhouettes, reflections, shadows, or background figures exist anywhere in the scene. The performer is the only living subject present at all times. He has orange-brown Shiba Inu fur, expressive canine features, a single black eyepatch over his right eye... His mouth opens and closes rhythmically in sync with the performance, with subtle head movement, gentle upper-body sway, and controlled side-to-side motion, creating a dynamic yet grounded stage presence. The microphone remains perfectly aligned with his mouth at all times, with no rotation. Lighting is moody and cinematic... The camera slowly pulls back and pans subtly to follow his movement while maintaining a full-body view, keeping both hands fully visible in-frame at all times."

**4. Musical Theater / Animated Characters**
For quirky, animated, or musical-theater-style songs, including the exact lyrics (even if you are providing an audio file) and describing the mouth mechanics helps precision:
> "A close-up of a cheerful girl puppet with curly auburn yarn hair and wide button eyes, holding a small red umbrella above her head. Rain falls gently around her. She looks upward and begins to sing with joy in English: "It's raining, it's raining, I love it when its raining." Her fabric mouth opening and closing to a melodic tune. Her hands grip the umbrella handle as she sways slightly from side to side in rhythm. The camera holds steady as the rain sparkles against the soft lighting. Her eyes blink occasionally as she sings."

**5. High-Emotion Singing with Gestures**
If the song builds to a crescendo, it helps to prompt for specific emotive body language alongside the singing:
> "A young woman sings with deep passion towards the camera, then slowly raises one hand to brush her hair back. She possesses long, voluminous dark brown wavy hair, deep blue eyes, and a sun-kissed complexion, wearing a rustic, textured dark burgundy off-shoulder top. She stands amidst a vast golden field, wildflowers swaying gently... The camera maintains a steady medium close-up, slightly low angle, focusing intently on her face. It subtly pushes in during her singing, then smoothly tracks her right hand as it rises, fingers lightly touching her temple, then fluidly sweeping the loose dark strands from her face, revealing her full expression. Intense golden backlighting creates a radiant halo around her hair..."

**6. Audio-Reactive Dancing (No Lip-Sync)**
If you are putting a song over the video but want the character to dance rather than sing, focus heavily on rhythmic tags:
> "A young woman with glowing eyes, crowned in black horns and adorned with intricate tattoos including wolves across her chest, slowly dances with passionate elegance under dim ambient light. Her dark hair flows as she sways, arms rising then falling rhythmically to unseen music; lips part slightly mid-motion. The camera glides smoothly around her from left to right, capturing her fluid grace against a misty, shadowed backdrop where faint snowflakes drift silently downward."
