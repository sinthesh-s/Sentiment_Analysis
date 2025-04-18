def set_background(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
        background_style = f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
            position: relative;
        }}

        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            backdrop-filter: blur(6px);
            background-color: rgba(0, 0, 0, 0.35);
            z-index: -1;
        }}

        @keyframes fadeSlide {{
            0% {{ opacity: 0; transform: translateY(-20px); }}
            100% {{ opacity: 1; transform: translateY(0); }}
        }}

        .main-container {{
            background: rgba(255, 255, 255, 0.15);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 20px;
            padding: 40px;
            max-width: 700px;
            margin: 80px auto;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(10px);
            animation: fadeSlide 0.8s ease-out;
        }}

        .stTextArea textarea {{
            background: rgba(255, 255, 255, 0.8);
            color: #333;
            border: none;
            border-radius: 12px;
            padding: 12px;
            font-size: 16px;
            transition: all 0.3s ease;
        }}

        .stTextArea textarea:focus {{
            border: 2px solid #ff4b4b;
            outline: none;
            background: rgba(255, 255, 255, 0.95);
        }}

        .stButton>button {{
            background: linear-gradient(145deg, #ff4b4b, #ff6b6b);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.6em 1.4em;
            font-weight: 600;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
        }}

        .stButton>button:hover {{
            background: white;
            color: #ff4b4b;
            border: 2px solid #ff4b4b;
            transform: translateY(-2px) scale(1.03);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        }}

        h1, .stMarkdown h1 {{
            color: #ffffff;
            text-align: center;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
        }}

        .stMarkdown p {{
            color: #eee;
            font-size: 16px;
            line-height: 1.6;
        }}
        </style>
        """
        st.markdown(background_style, unsafe_allow_html=True)
