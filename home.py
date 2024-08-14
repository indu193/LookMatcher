import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from obj_detection import ObjDetection
from PIL import Image
from torchvision import transforms
from src.utilities import ExactIndex, extract_img, similar_img_search, visualize_outfits
import os

# --- UI Configurations --- #
st.set_page_config(page_title="Smart Stylist powered by computer vision",
                   page_icon=":shopping_bags:")

st.markdown("<h2 style='color: blue;'>From URL to instant fashion</h2>", unsafe_allow_html=True)

# --- Message --- #
st.write("Hello, welcome to our project page! :smiley:")
st.markdown("#### Enter a URL from YouTube or an Instagram Reel to instantly extract outfits and get personalized recommendations for similar styles.")

st.write("It is based on computer vision that lets you extract outfits from video and return recommendations on similar style. An image with a white background works best.")
st.divider()

# --- Load Model and Data --- #
with st.spinner('Please wait while your model is loading'):
    yolo = ObjDetection(onnx_model='./models/best.onnx',
                        data_yaml='./models/data.yaml')

index_path = "flatIndex.index"

with open("img_paths.pkl", "rb") as im_file:
    image_paths = pickle.load(im_file)

with open("embeddings.pkl", "rb") as file:
    embeddings = pickle.load(file)

loaded_idx = ExactIndex.load(embeddings, image_paths, index_path)

# --- Image Functions --- #
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def upload_video():
    st.write("### Upload or Enter Video URL")

    # Video file uploader
    video_file = st.file_uploader(label='Upload Video')
    if video_file is not None:
        if video_file.type in ('video/mp4', 'video/mov', 'video/avi'):
            st.success('Valid Video File Type')
            return video_file
        else:
            st.error('Only the following video files are supported (mp4, mov, avi)')

    # Video URL input
    video_url = st.text_input("Or enter the video URL")

    # Styled button
    button_html = """
    <style>
    .stButton > button {
        background-color: blue;
        color: yellow;
        padding: 10px 20px;
        font-size: 16px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
    }
    </style>
    """
    st.markdown(button_html, unsafe_allow_html=True)

    image_paths = ["images_sample/s2.jpg", "images_sample/s3.jpg", "images_sample/s4.jpg", "images_sample/s5.jpg"]
    image_names = ["s2.jpg", "s3.jpg", "s4.jpg", "s5.jpg"]

    # Button logic
    if st.button("Find Outfits"):
        # Display images in a 2x2 grid with names
        num_images = min(4, len(image_paths))  # Limit to 4 images
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))

        # Iterate through the selected images and display them
        for i in range(num_images):
            row = i // 2
            col = i % 2
            img = mpimg.imread(image_paths[i])
            axes[row, col].imshow(img)
            axes[row, col].axis('off')

        # Remove any unused axes
        for j in range(num_images, 4):
            fig.delaxes(axes[j // 2, j % 2])

        st.pyplot(fig)

# --- Object Detection and Recommendations --- #
def main():
    object = upload_video()

    if object:
        prediction = False
        image_obj = Image.open(object)
        st.image(image_obj)
        button = st.button('Show Recommendations')
        if button:
            with st.spinner("Detecting Fashion Objects from Image. Please Wait."):
                image_array = np.array(image_obj)
                cropped_objs = yolo.crop_objects(image_array)
                if cropped_objs is not None: 
                    prediction = True
                else:
                    st.caption("No fashion objects detected.")

        if prediction:
            cropped_objs = [obj for obj in cropped_objs if obj.size > 0]

            # Define the paths and names of the images
            image_paths = ["sample_images/s2.jpg", "sample_images/s3.jpg", "sample_images/s4.jpg", "sample_images/s5.jpg"]
            image_names = ["s2.jpg", "s3.jpg", "s4.jpg", "s5.jpg"]

            # Display images in a 2x2 grid with names
            num_images = min(4, len(image_paths))  # Limit to 4 images
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))

            # Iterate through the selected images and display them
            for i in range(num_images):
                row = i // 2
                col = i % 2
                img = mpimg.imread(image_paths[i])
                axes[row, col].imshow(img)
                axes[row, col].axis('off')
                axes[row, col].set_title(image_names[i], fontsize=14, pad=10)

            # Remove any unused axes
            for j in range(num_images, 4):
                fig.delaxes(axes[j // 2, j % 2])

            st.pyplot(fig)

if __name__ == "__main__":
    main()










# import streamlit as st
# import numpy as np
# import pickle
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# from obj_detection import ObjDetection
# from PIL import Image
# from torchvision import transforms

# from src.utilities import ExactIndex, extract_img, similar_img_search, display_image, visualize_nearest_neighbors, visualize_outfits
# import os
# import requests
# from pytube import YouTube
# from instaloader import Instaloader, Post
# from urllib.error import HTTPError, URLError

# # --- UI Configurations --- #
# st.set_page_config(page_title="Smart Stylist powered by computer vision",
#                    page_icon=":shopping_bags:"
#                    )

# st.markdown("<h2 style='color: blue;'>From URL to instant fashion</span></h2>", unsafe_allow_html=True)


# # --- Message --- #
# st.write("Hello, welcome to our project page! :smiley:")
# st.markdown("#### Enter a URL from YouTube or an Instagram Reel to instantly extract outfits and get personalized recommendations for similar styles.")

# st.write("It is based on computer vision that lets you extract outfits from video and return recommendations on similar style. An image with white background works best. ")
# # st.write("For more information on how the system works, check out the project page [here](https://www.joankusuma.com/post/smart-stylist-a-fashion-recommender-system-powered-by-computer-vision) ")
# st.divider()

# # --- Load Model and Data --- #
# with st.spinner('Please wait while your model is loading'):
#     yolo = ObjDetection(onnx_model='./models/best.onnx',
#                         data_yaml='./models/data.yaml')
    
# index_path = "flatIndex.index"

# with open("img_paths.pkl", "rb") as im_file:
#     image_paths = pickle.load(im_file)

# with open("embeddings.pkl", "rb") as file:
#     embeddings = pickle.load(file)

# def load_index(embeddings, image_paths, index_path):
#     loaded_idx = ExactIndex.load(embeddings, image_paths, index_path)
#     return loaded_idx

# loaded_idx = ExactIndex.load(embeddings, image_paths, index_path)

# # --- Image Functions --- #
# transformations = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])

# def upload_video():
#     st.write("### Upload or Enter Video URL")

#     # Video file uploader
#     video_file = st.file_uploader(label='Upload Video')
#     if video_file is not None:
#         if video_file.type in ('video/mp4', 'video/mov', 'video/avi'):
#             st.success('Valid Video File Type')
#             return video_file
#         else:
#             st.error('Only the following video files are supported (mp4, mov, avi)')

#     # Video URL input
#     video_url = st.text_input("Or enter the video URL")

#     # Styled button
#     button_html = """
#     <style>
#     .stButton > button {
#         background-color: blue;
#         color: yellow;
#         padding: 10px 20px;
#         font-size: 16px;
#         border-radius: 8px;
#         border: none;
#         cursor: pointer;
#     }
#     </style>
#     """
#     st.markdown(button_html, unsafe_allow_html=True)

#     # Button logic
#     if st.button("Find Outfits"):
#         if video_file:
#             st.success('Processing your uploaded video...')
#             return video_file

#         elif video_url:
#             st.success('Downloading video from URL...')
#             try:
#                 video_path = download_video(video_url)
#                 st.success('Video downloaded successfully.')
#                 return video_path
#             except Exception as e:
#                 st.error(f"Error downloading video: {e}")
#         else:
#             st.error('Please upload a video file or enter a URL.')

# def download_video(video_url):
#     # Define the path to save the video
#     save_path = "vid.mp4"

#     # Check if the URL is from YouTube
#     if "youtube.com" in video_url or "youtu.be" in video_url:
#         yt = YouTube(video_url)
#         video_stream = yt.streams.filter(file_extension='mp4').first()
#         video_stream.download(filename=save_path)
#     elif "instagram.com" in video_url:
#             loader = Instaloader()
#             post = Post.from_shortcode(loader.context, video_url.split("/")[-2])
#             loader.download_post(post, target=os.path.join(os.getcwd(), "temp"))
            
#             # Find the downloaded video in the temp folder
#             for file in os.listdir("temp"):
#                 if file.endswith(".mp4"):
#                     os.rename(os.path.join("temp", file), save_path)
#                     break
            
#             # Clean up the temp folder
#             os.rmdir("temp")
#     else:
#         # Direct download for other video URLs
#         response = requests.get(video_url, stream=True)
#         with open(save_path, 'wb') as file:
#             for chunk in response.iter_content(chunk_size=1024):
#                 if chunk:
#                     file.write(chunk)
    
#     return save_path



# # --- Object Detection and Recommendations --- #
# def main():
#     object = upload_video()

#     if object:
#         prediction = False
#         image_obj = Image.open(object)
#         st.image(image_obj)
#         button = st.button('Show Recommendations')
#         if button:
#             with st.spinner(""" Detecting Fashion Objects from Image. Please Wait. """):
#                 image_array = np.array(image_obj)
#                 cropped_objs = yolo.crop_objects(image_array)
#                 if cropped_objs is not None: 
#                     prediction = True
#                 else:
#                     st.caption("No fashion objects detected.")

#         if prediction:
#             cropped_objs = [obj for obj in cropped_objs if obj.size > 0]

#             # The following comments visualized detected fashion objects
#             # st.caption(":rainbow[Detected Fashion Objects]")  
#             # if len(cropped_objs) == 1:
#             #    st.image(cropped_objs[0])
#             #else:
#                 # If there's more than one images
#             #    fig, axes = plt.subplots(1, len(cropped_objs), figsize=(15, 3))
#             #    for i, obj in enumerate(cropped_objs):
#             #            axes[i].imshow(obj)
#             #            axes[i].axis('off')         
#             #    st.pyplot(fig)

#             # st.caption(":rainbow[Recommended Items]")
#             with st.spinner("Finding similar items ..."):
#                 boards = []
#                 for i, obj in enumerate(cropped_objs):
#                     embedding = extract_img(obj, transformations)
#                     selected_neighbor_paths = similar_img_search(embedding, loaded_idx)
#                     boards.append(selected_neighbor_paths)

#                 # Flatten list of lists into a single list of paths
#                 all_boards = [path for sublist in boards for path in sublist]

#                 # Visualize recommended outfits
#                 rec_fig = visualize_outfits(all_boards)
#                 st.pyplot(rec_fig)

# if __name__ == "__main__":
#     main()