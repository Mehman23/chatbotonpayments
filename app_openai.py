
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
from datetime import datetime
import os

import streamlit as st
import logging


my_key_openai = st.secrets["mykey_openai"]


llm_openai = ChatOpenAI(api_key = my_key_openai, model = "gpt-4o-mini", temperature=0.2, streaming=True)
embeddings = OpenAIEmbeddings(api_key = my_key_openai, model="text-embedding-3-large")


logging.basicConfig(filename='error_log.csv', level=logging.ERROR, 
                    format='%(asctime)s,%(levelname)s,%(message)s')

st.set_page_config(page_title="AI Lawyer Chatbot", page_icon="🤖", layout="centered")

if "user_info_submitted" not in st.session_state:
    st.session_state.user_info_submitted = False

if not st.session_state.user_info_submitted:

    with st.form("user_info_form"):
        st.header("Lawyer on Payments Chatbot 💬")
        st.write("Məlumatlarızı daxil edin")
        
        name = st.text_input("Ad")
        surname = st.text_input("Soyad")
        department = st.text_input("Departament adı")

        submit_button = st.form_submit_button("Daxil ol")
        
        if submit_button:
            if name and surname and department:
            
                st.session_state.user_info = {"name": name, "surname": surname, "department": department}
                st.session_state.user_info_submitted = True
                st.success("Məlumatlar qeyd edildi")
                st.rerun()

            else:
                st.warning("Məlumatlarızı daxil edin")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("Created by **Azərbaycan Respublikasının Mərkəzi Bankı** © 2024", unsafe_allow_html=True)

else:
    st.title("💬 AI Lawyer Chatbot")
    st.divider()


    def get_final_prompt(prompt):

        user_messages = [msg["content"] for msg in reversed(st.session_state.messages) if msg["role"] == "user"]

        if len(user_messages) >= 2:
            previous_prompt = user_messages[1]
            latest_prompt = user_messages[0]
            full_query = previous_prompt + ". " + latest_prompt
        else:
            full_query = user_messages[0] if user_messages else ""

        try:

            new_vector_store = FAISS.load_local(
                "faiss_index_numeric_openai", embeddings, allow_dangerous_deserialization=True
                )

            retriever = new_vector_store.as_retriever()

            relevant_documents = retriever.invoke(full_query)

        except Exception as e:

            logging.error(f"Error loading FAISS index: {e}")

        context_data = " ".join([document.page_content for document in relevant_documents])

        final_prompt = f"""
        Sənə bir sual verəcəm və cavablandırmaq üçün Ödəniş xidmətlərinin fəaliyyəti çərçivəsində Azərbaycan Respublikasının qanunvericiliyinə aid dörd fərqli məlumat təqdim edəcəm. Məlumatlar 'Başlıq: məlumat' strukturundadır. Başlıq hər məlumatın hansı qanuna və maddəyə aid olduğunu bildirir. 
        
        Cavab hazırlayarkən bu təlimatlara diqqət et:
        
        - Əgər verilən sual ilə təqdim edilən məlumatların mövzusu uyğun deyilsə, 'Bağışlayın,bu haqda dəqiq məlumatım yoxdur. Ya da sizi səhv başa düşdüm. Sualınızı daha ətraflı yazmanızı xahiş edirəm. Mən sizə ödəniş xidmətlərinin fəaliyyəti çərçivəsində köməklik göstərə bilərəm.' bənzər cavablar ver!
        - Cavabını sadə və aydın dildə yaz.
        - Gərəklidirsə, izahını nümunələrlə dəstəklə.
        - Lazım olduqda, istinad etdiyin qanun və maddələr haqqında yaz.
        - Cavabını ətraflı, lakin konkret və dəqiq yaz.
        - Cavabında sənə məlumatlar verildiyindən bəhs etmə. Sənə məlumat təqdim edildiyini yazma.

        Sual: {prompt}
        Məlumatlar: {context_data}
        """
        return final_prompt

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "system", "content": "sən, hüquqi mövzuları sadə dildə izah edən robotsan."})

    for message in st.session_state.messages[1:]:
        role = message.get("role", "user")
        avatar = message.get("avatar", "👨🏻" if role == "user" else "🤖")
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message.get("content", ""))

    if prompt := st.chat_input("Sualınızı yazın..."):
        
        st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "👨🏻"})
        st.chat_message("user", avatar="👨🏻").markdown(prompt)

        response_text = ""
        with st.chat_message("assistant", avatar="🤖") as assistant_message:
            response_placeholder = st.empty() 

            try:

                for token in llm_openai.stream(st.session_state.messages + [{"role": "user", "content": get_final_prompt(prompt)}]):
                    token_content = token.content
                    response_text += token_content
                    response_placeholder.markdown(response_text) 

            except Exception as ex:

                logging.error(f"Error streaming response: {ex}")
                st.error("Cavab verərkən bir xəta baş verdi. Zəhmət olmasa sonra yenidən cəhd edin.")

            st.session_state.messages.append({"role": "assistant", "content": response_text, "avatar": "🤖"})

    feedback_pos_file = "feedback_pos.csv"
    feedback_neg_file = "feedback_neg.csv"

    if len(st.session_state.messages) > 2:

        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ düzgün cavab"):

                try:

                    df_pos = pd.DataFrame({"user_name": st.session_state.user_info["name"],
                                        "user_surname": st.session_state.user_info["surname"],
                                        "user_department": st.session_state.user_info["department"],
                                        "respond": [st.session_state.messages], 
                                        "feedback": ["thumbs up"], 
                                        "user_comment": [""],
                                        "timestamp": datetime.now()})
                    df_pos.to_csv(feedback_pos_file, mode='a', header=not os.path.exists(feedback_pos_file), index=False, encoding='utf-8-sig')
                    st.success("Rəy alındı.")

                except Exception as ex_pos:

                    logging.error(f"Error streaming response: {ex_pos}")
                    st.error("Rəy yadda saxlanarkən bir xəta baş verdi. Zəhmət olmasa sonra yenidən cəhd edin.")

        with col2:
            if st.button("❌ yanlış cavab"):
                st.session_state.feedback = "thumbs down"
                
            if st.session_state.get("feedback") == "thumbs down":
                feedback_text = st.text_input("Təsvir edin")

                if st.button("Təsviri yadda saxla"):

                    try:

                        df_neg = pd.DataFrame({"user_name": st.session_state.user_info["name"],
                                            "user_surname": st.session_state.user_info["surname"],
                                            "user_department": st.session_state.user_info["department"],
                                            "respond": [st.session_state.messages], 
                                            "feedback": ["thumbs down"], 
                                            "user_comment": [feedback_text],
                                            "timestamp": datetime.now()})
                        df_neg.to_csv(feedback_neg_file, mode='a', header=not os.path.exists(feedback_neg_file), index=False, encoding='utf-8-sig')
                        st.success("Rəy alındı.")
                        st.session_state.feedback = None

                    except Exception as ex_im:

                        logging.error(f"Error streaming response: {ex_im}")
                        st.error("Təsvir yadda saxlanarkən bir xəta baş verdi. Zəhmət olmasa sonra yenidən cəhd edin.")
                
                elif feedback_text:

                    try:

                        df_neg = pd.DataFrame({"user_name": st.session_state.user_info["name"],
                                            "user_surname": st.session_state.user_info["surname"],
                                            "user_department": st.session_state.user_info["department"],
                                            "respond": [st.session_state.messages],
                                            "feedback": ["thumbs down"],
                                            "user_comment": [feedback_text],
                                            "timestamp": datetime.now()})
                        df_neg.to_csv(feedback_neg_file, mode='a', header=not os.path.exists(feedback_neg_file), index=False, encoding='utf-8-sig')
                        st.success("Rəy alındı.")
                        st.session_state.feedback = None

                    except Exception as ex_im_2:

                        logging.error(f"Error streaming response: {ex_im_2}")
                        st.error("Təsvir yadda saxlanarkən bir xəta baş verdi. Zəhmət olmasa sonra yenidən cəhd edin.")

                
                elif st.button("Təsvir etmək istəmirəm"):

                    try:

                        df_neg = pd.DataFrame({"user_name": st.session_state.user_info["name"],
                                            "user_surname": st.session_state.user_info["surname"],
                                            "user_department": st.session_state.user_info["department"],
                                            "respond": [st.session_state.messages], 
                                            "feedback": ["thumbs down"], 
                                            "user_comment": [""],
                                            "timestamp": datetime.now()})
                        df_neg.to_csv(feedback_neg_file, mode='a', header=not os.path.exists(feedback_neg_file), index=False, encoding='utf-8-sig')
                        st.success("Rəy alındı.")
                        st.session_state.feedback = None

                    except Exception as ex_im_2:

                        logging.error(f"Error streaming response: {ex_im_2}")
                        st.error("Təsvirdən imtina edərkən bir xəta baş verdi. Zəhmət olmasa sonra yenidən cəhd edin.")


                

