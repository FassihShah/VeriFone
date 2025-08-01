import streamlit as st
import tempfile
import uuid
import os
import time
import shutil
import atexit
from agent.graph import supervisor_agent, system_prompt

st.set_page_config(
    page_title="📱 Mobile Assistant",
    layout="centered",
    initial_sidebar_state="auto"
)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]

if "processed_images" not in st.session_state:
    st.session_state.processed_images = set()

if "temp_files" not in st.session_state:
    st.session_state.temp_files = []

if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp(prefix=f"mobile_assistant_{st.session_state.thread_id}_")
    print(f"Created temp directory: {st.session_state.temp_dir}")


def create_persistent_temp_file(image_bytes, filename):
    """Saves image bytes to a persistent temp file for the session."""
    temp_dir = st.session_state.temp_dir
    safe_filename = os.path.basename(filename)
    timestamp = int(time.time() * 1000)
    temp_filename = f"mobile_image_{timestamp}_{safe_filename}"
    temp_path = os.path.join(temp_dir, temp_filename)

    with open(temp_path, 'wb') as f:
        f.write(image_bytes)

    st.session_state.temp_files.append(temp_path)
    return temp_path.replace('\\', '/')

def cleanup_temp_dir():
    """Recursively deletes the session's temporary directory and all its contents."""
    temp_dir = st.session_state.get("temp_dir")
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temp directory and all its contents: {temp_dir}")
            st.session_state.temp_files = []
            st.session_state.processed_images = set()
            if "temp_dir" in st.session_state:
                del st.session_state.temp_dir
        except Exception as e:
            print(f"Error cleaning up temp directory {temp_dir}: {e}")

atexit.register(cleanup_temp_dir)

def handle_uploaded_image(uploaded_image_file):
    """Processes an uploaded image, calls the agent, and updates the chat."""
    if uploaded_image_file:
        image_id = f"{uploaded_image_file.name}_{uploaded_image_file.size}"

        if image_id not in st.session_state.processed_images:
            st.session_state.processed_images.add(image_id)
            image_bytes = uploaded_image_file.getvalue()

            try:
                with st.spinner("🔍 Analyzing image..."):
                    image_path = create_persistent_temp_file(image_bytes, uploaded_image_file.name)
                    user_msg = (
                        "I've uploaded an image of a mobile phone. Please analyze it to detect "
                        f"the brand and model. The image is at: {image_path}"
                    )
                    st.session_state.messages.append({"role": "user", "content": user_msg})

                    result = supervisor_agent.invoke(
                        {"messages": st.session_state.messages},
                        config={"configurable": {"thread_id": st.session_state.thread_id}}
                    )

                print(result)
                assistant_content = result['messages'][-1].content if result and 'messages' in result else "Sorry, I couldn't process the image."
                st.session_state.messages.append({"role": "assistant", "content": assistant_content})
                st.rerun()

            except Exception as e:
                st.error(f"Error processing image: {e}")
                if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                    st.session_state.messages.pop()

with st.sidebar:
    st.markdown("## 📎 Upload an Image")
    uploaded_image = st.file_uploader(
        "Upload a photo of a mobile phone for analysis.",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_image:
        st.image(uploaded_image, caption="Image for Analysis", use_container_width=True)

    st.markdown("---")
    st.markdown("## 🛠️ Options")

    if st.button("🗑️ Clear Chat & Files", use_container_width=True):
        cleanup_temp_dir()
        st.session_state.messages = [{"role": "system", "content": system_prompt}]
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.markdown("---")
    st.markdown("## 💡 Tips")
    st.info(
        """
        - Upload clear back-side photos.
        - Specify condition (e.g., '10/10').
        - Mention PTA approval status.
        - Include RAM/Storage if known.
        """
    )
    if st.session_state.temp_files:
        with st.expander("Show temp files for debug"):
            st.write(st.session_state.temp_files)


st.title("📱 Mobile Assistant")
st.caption("Your go-to expert for mobile specs, prices, and photo analysis")

for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        if msg["role"] == "user" and "I've uploaded an image" in msg["content"]:
            st.markdown("🖼️ *Analyzed the uploaded image.*")
        else:
            st.markdown(msg["content"])
handle_uploaded_image(uploaded_image)

if prompt := st.chat_input("Ask anything about phones..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    try:
        with st.spinner("🤔 Thinking..."):
            result = supervisor_agent.invoke(
                {"messages": st.session_state.messages},
                config={"configurable": {"thread_id": st.session_state.thread_id}}
            )

        print(result)

        assistant_content = result['messages'][-1].content if result and 'messages' in result else "Sorry, I couldn't process your request."
        st.session_state.messages.append({"role": "assistant", "content": assistant_content})
        st.rerun() 

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            st.session_state.messages.pop()