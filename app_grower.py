import os
import streamlit as st
import base64
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from PIL import Image
from langchain_core.messages import HumanMessage
import io

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Master Grower AI",
    page_icon="🌿",
    layout="centered"
)

# Cargar variables de entorno
# Cargar variables de entorno locales (si existen)
load_dotenv()

# Lógica híbrida: Intenta leer de la Nube (Secrets), si falla, lee de Local (.env)
try:
    CLAVE_API_GOOGLE = st.secrets["GOOGLE_API_KEY"]
except:
    CLAVE_API_GOOGLE = os.getenv("GOOGLE_API_KEY")

# Verificación de seguridad
if not CLAVE_API_GOOGLE:
    st.error("❌ No se encontró la API KEY. Configúrala en el archivo .env (Local) o en Secrets (Nube).")
    st.stop()
MODELO_EMBEDDING_LOCAL = "sentence-transformers/all-MiniLM-L6-v2"
CARPETA_BASE_DATOS = "cerebro_cultivo_faiss"

# --- TÍTULO Y BARRA LATERAL ---
st.title("🌿 Master Grower AI")
st.caption("Tu asistente experto en cultivo, botánica y biopreparados.")

with st.sidebar:
    st.header("⚙️ Panel de Control")
    if st.button("🗑️ Borrar Historial"):
        st.session_state.messages = []
        st.rerun()
    st.info("Base de datos cargada: 1700+ páginas")
    
    st.divider()
    st.header("📸 Ojos del Grower")
    imagen_subida = st.file_uploader("Sube una foto de tu planta", type=["jpg", "png", "jpeg"])
    
    if imagen_subida:
        # Mostramos la imagen en pantalla
        image = Image.open(imagen_subida)
        # CORRECCIÓN AQUÍ: Se cambió use_column_width por use_container_width
        st.image(image, caption="Imagen cargada", use_container_width=True)
        # ... (código anterior de la barra lateral) ...
    
    st.divider()
    st.header("🛠️ Diagnóstico de Nube")
    if st.button("Ver Modelos Disponibles"):
        try:
            import google.generativeai as genai
            genai.configure(api_key=CLAVE_API_GOOGLE)
            st.write("Conectando con Google...")
            modelos = genai.list_models()
            lista_modelos = []
            for m in modelos:
                if 'generateContent' in m.supported_generation_methods:
                    lista_modelos.append(m.name)
            st.success("✅ Modelos detectados:")
            st.code(lista_modelos)
        except Exception as e:
            st.error(f"Error de conexión: {e}")

# --- FUNCIÓN DE CARGA ---
@st.cache_resource
def cargar_cerebro():
    if not os.path.exists(CARPETA_BASE_DATOS):
        st.error(f"❌ No encontré la carpeta '{CARPETA_BASE_DATOS}'. Ejecuta primero el script de ingesta.")
        return None
    try:
        embeddings = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDING_LOCAL)
        db = FAISS.load_local(CARPETA_BASE_DATOS, embeddings, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Error cargando la base de datos: {e}")
        return None

db = cargar_cerebro()

# --- GESTIÓN DEL CHAT ---
# 1. Inicializar historial
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Mostrar historial en pantalla
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. BARRA DE CHAT (Lógica Principal)
if prompt := st.chat_input("¿Qué necesita tu cultivo hoy?"):
    # Guardar y mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar respuesta de la IA
    if db:
        with st.chat_message("assistant"):
            with st.spinner("El Master Grower está analizando (Texto + Visión)..."):
                try:
                    # A. Configuración del Modelo (1.5 Flash para visión)
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash-lite-preview-09-2025",
                        google_api_key=CLAVE_API_GOOGLE,
                        temperature=0.3
                    )

                    # B. Búsqueda en Libros (RAG)
                    docs = db.similarity_search(prompt, k=4)
                    contexto_texto = "\n\n".join([d.page_content for d in docs])

                    # C. Prompt del Sistema
                    system_prompt = """
                    Actúa como un Master Grower experto.
                    CONTEXTO GEOGRÁFICO: Estás en Argentina (Hemisferio Sur). Invierte las estaciones del Hemisferio Norte (ej. Marzo=Cosecha, Septiembre=Siembra).
                    
                    Tu misión:
                    1. Si hay una imagen, ANALÍZALA buscando plagas, deficiencias o estado de floración.
                    2. Usa el CONTEXTO de los libros para dar una solución científica y orgánica.
                    3. Si la imagen es sana, felicita al cultivador.
                    """

                    # D. Ejecución (Texto o Multimodal)
                    if imagen_subida:
                        # --- MODO VISIÓN ---
                        st.info("👁️ Analizando imagen adjunta...")
                        
                        # 1. Resetear el puntero del archivo
                        imagen_subida.seek(0)
                        
                        # 2. Convertir la imagen a Base64
                        bytes_data = imagen_subida.getvalue()
                        image_b64 = base64.b64encode(bytes_data).decode()
                        
                        # 3. Crear la URI de la imagen
                        image_url = f"data:{imagen_subida.type};base64,{image_b64}"

                        mensaje_multimodal = [
                            {
                                "type": "text",
                                "text": f"{system_prompt}\n\nCONTEXTO DE LIBROS:\n{contexto_texto}\n\nPREGUNTA USUARIO: {prompt}"
                            },
                            {
                                "type": "image_url",
                                "image_url": image_url
                            }
                        ]
                        respuesta_ai = llm.invoke([HumanMessage(content=mensaje_multimodal)])
                        respuesta_texto = respuesta_ai.content
                    else:
                        # --- MODO TEXTO ---
                        msg = f"{system_prompt}\n\nCONTEXTO:\n{contexto_texto}\n\nPREGUNTA: {prompt}"
                        respuesta_ai = llm.invoke(msg)
                        respuesta_texto = respuesta_ai.content

                    # E. Mostrar Respuesta
                    st.markdown(respuesta_texto)

                    # Mostrar fuentes
                    with st.expander("🔎 Referencias Bibliográficas"):
                        for doc in docs:
                            origen = os.path.basename(doc.metadata.get('source', 'Doc'))
                            st.caption(f"📖 {origen}")

                    # Guardar respuesta en historial
                    st.session_state.messages.append({"role": "assistant", "content": respuesta_texto})

                except Exception as e:

                    st.error(f"Ocurrió un error: {e}")







