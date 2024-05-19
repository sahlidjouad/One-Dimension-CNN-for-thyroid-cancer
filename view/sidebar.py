
import streamlit as st



def config_sidebar():
    with st.sidebar:
        
        st.info("**Yo fam! Start here ↓**", icon="👋🏾")
            
        with st.expander(":rainbow[**Refine your output here**]"):
            # Advanced Settings (for the curious minds!)
            width = st.number_input("Width of output image", value=1024)
            height = st.number_input("Height of output image", value=1024)
            num_outputs = st.slider("Number of images to output", value=1, min_value=1, max_value=4)
            scheduler = st.selectbox('Scheduler', ('DDIM', 'DPMSolverMultistep', 'HeunDiscrete','KarrasDPM', 'K_EULER_ANCESTRAL', 'K_EULER', 'PNDM'))
            num_inference_steps = st.slider("Number of denoising steps", value=50, min_value=1, max_value=500)
            guidance_scale = st.slider("Scale for classifier-free guidance", value=7.5, min_value=1.0, max_value=50.0, step=0.1)
            prompt_strength = st.slider("Prompt strength when using img2img/inpaint(1.0 corresponds to full destruction of infomation in image)", value=0.8, max_value=1.0, step=0.1)
            refine = st.selectbox("Select refine style to use (left out the other 2)", ("expert_ensemble_refiner", "None"))
            high_noise_frac = st.slider("Fraction of noise to use for `expert_ensemble_refiner`", value=0.8, max_value=1.0, step=0.1)
                
        return width, height, num_outputs, scheduler, num_inference_steps, guidance_scale, prompt_strength, refine, high_noise_frac

