def set_background(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
        background_style = f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
        }}

        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5);  /* Slight dark overlay */
            z-index: -1;
        }}

        @keyframes fadeSlide {{
            0% {{ opacity: 0; transform: translateY(-20px); }}
            100% {{ opacity: 1; transform: translateY(0); }}
        }}

        .main-container {{
            background: rgba(255, 255, 255, 0.85);  /* Light background for readability */
            border-radius: 16px;
            padding: 30px;
            max-width: 600px;
            margin: 100px auto;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            animation: fadeSlide 0.8s ease-out;
        }}

        .stTextArea textarea {{
            background-color: white;
            color: black;
            border-radius: 8px;
            border: 1px solid #ccc;
        }}

        .stButton>button {{
            background-color: #ff4b4b;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5em 1em;
            font-weight: bold;
            transition: 0.3s ease;
        }}

        .stButton>button:hover {{
            background-color: white;
            color: #ff4b4b;
            border: 1px solid #ff4b4b;
            transform: scale(1.05);
        }}

        h1, .stMarkdown {{
            color: #111;
            text-align: center;
        }}
        </style>
        """
        st.markdown(background_style, unsafe_allow_html=True)
