// Data_Collector/automind-firestore.js
//
// Recibe AutoMind_Info desde AutoMindCloud/__init__.py y escribe:
//
// AutoMind_Data_DD-MM-AAAA
// └── IP_xxx.xxx.xxx.xxx
//     └── JSON
//         └── documento automático
//
// El documento JSON contiene:
//   - AutoMind_Info
//   - User_Info
//   - timestamp_servidor
//
// La ruta coincide exactamente con firestore.rules.

import {
  initializeApp,
  getApps,
  getApp
} from "https://www.gstatic.com/firebasejs/9.22.0/firebase-app.js";

import {
  getFirestore,
  collection,
  addDoc,
  serverTimestamp,
  waitForPendingWrites
} from "https://www.gstatic.com/firebasejs/9.22.0/firebase-firestore.js";

import {
  getAuth,
  signInAnonymously
} from "https://www.gstatic.com/firebasejs/9.22.0/firebase-auth.js";


// ============================================================
// CONFIGURACIÓN FIREBASE
// ============================================================
const FIREBASE_CONFIG = Object.freeze({
  apiKey: "AIzaSyBSC-OGbSo_8wJlv9nSLJ8lUojcEKimOBQ",
  authDomain: "automindrobotics.firebaseapp.com",
  projectId: "automindrobotics",
  storageBucket: "automindrobotics.firebasestorage.app",
  messagingSenderId: "619255898589",
  appId: "1:619255898589:web:24605a66f71f9f9ae71634"
});

// Debe ser exactamente el ID de la base Firestore donde publicas las reglas.
// Si tu base es la predeterminada, cambia esta línea a "(default)".
const DATABASE_ID = "automindcolab";

const APP_NAME = "automind-colab-firestore";
const IP_ENDPOINT = "https://api.ipify.org?format=json";
const IP_TIMEOUT_MS = 8000;


// ============================================================
// FECHA
// ============================================================
function fechaHoraConsulta() {
  const ahora = new Date();

  const dia = String(ahora.getDate()).padStart(2, "0");
  const mes = String(ahora.getMonth() + 1).padStart(2, "0");
  const anio = String(ahora.getFullYear());
  const hora = String(ahora.getHours()).padStart(2, "0");
  const minuto = String(ahora.getMinutes()).padStart(2, "0");
  const segundo = String(ahora.getSeconds()).padStart(2, "0");

  return `${dia}-${mes}-${anio}-${hora}:${minuto}:${segundo}`;
}


function fechaParaColeccion() {
  const ahora = new Date();

  const dia = String(ahora.getDate()).padStart(2, "0");
  const mes = String(ahora.getMonth() + 1).padStart(2, "0");
  const anio = String(ahora.getFullYear());

  return `${dia}-${mes}-${anio}`;
}


function fechaValida(fecha) {
  return (
    typeof fecha === "string" &&
    /^\d{2}-\d{2}-\d{4}$/.test(fecha)
  );
}


// ============================================================
// USER_INFO: DATOS EXPUESTOS POR EL NAVEGADOR
// ============================================================
function sistemaOperativoAproximado() {
  const ua = navigator.userAgent || "";

  if (ua.includes("Windows NT 10.0")) return "Windows 10/11";
  if (ua.includes("Windows NT 6.3")) return "Windows 8.1";
  if (ua.includes("Windows NT 6.2")) return "Windows 8";
  if (ua.includes("Windows NT 6.1")) return "Windows 7";
  if (ua.includes("Mac OS X")) return "macOS";
  if (ua.includes("Android")) return "Android";
  if (/iPhone|iPad|iPod/.test(ua)) return "iOS";
  if (ua.includes("Linux")) return "Linux";

  return "Desconocido";
}


function arquitecturaDesdeUserAgent(noDisponible) {
  const ua = navigator.userAgent || "";

  if (/ARM64|aarch64/i.test(ua)) return "ARM64";
  if (/Win64|x86_64|x64|amd64/i.test(ua)) return "x86_64";
  if (/i[3-6]86|x86/i.test(ua)) return "x86";

  return noDisponible;
}


async function arquitectura(noDisponible) {
  try {
    if (navigator.userAgentData?.getHighEntropyValues) {
      const datos = await navigator.userAgentData.getHighEntropyValues([
        "architecture",
        "bitness"
      ]);

      const arch = String(datos.architecture || "").trim();
      const bits = String(datos.bitness || "").trim();

      const resultado = [
        arch,
        bits ? `${bits}-bit` : null
      ]
        .filter(Boolean)
        .join(" ");

      if (resultado) {
        return resultado;
      }
    }
  } catch (_) {
    // Usa el fallback basado en userAgent.
  }

  return arquitecturaDesdeUserAgent(noDisponible);
}


function gpu(noDisponible) {
  try {
    const canvas = document.createElement("canvas");
    const gl =
      canvas.getContext("webgl") ||
      canvas.getContext("experimental-webgl");

    if (!gl) {
      return noDisponible;
    }

    const extension = gl.getExtension("WEBGL_debug_renderer_info");

    if (!extension) {
      return noDisponible;
    }

    const vendor = gl.getParameter(extension.UNMASKED_VENDOR_WEBGL);
    const renderer = gl.getParameter(extension.UNMASKED_RENDERER_WEBGL);

    return vendor || renderer
      ? `${vendor || "Sin vendor"} | ${renderer || "Sin renderer"}`
      : noDisponible;

  } catch (_) {
    return noDisponible;
  }
}


function red(noDisponible) {
  const conexion =
    navigator.connection ||
    navigator.mozConnection ||
    navigator.webkitConnection;

  if (!conexion) {
    return {
      tipo: noDisponible,
      latencia: noDisponible,
      anchoBanda: noDisponible
    };
  }

  return {
    tipo: conexion.effectiveType || noDisponible,
    latencia:
      typeof conexion.rtt === "number"
        ? `${conexion.rtt} ms`
        : noDisponible,
    anchoBanda:
      typeof conexion.downlink === "number"
        ? `${conexion.downlink} Mbps`
        : noDisponible
  };
}


function pantalla(noDisponible) {
  try {
    if (typeof screen === "undefined") {
      return noDisponible;
    }

    const ancho = screen.width || noDisponible;
    const alto = screen.height || noDisponible;
    const escala = window.devicePixelRatio || 1;

    return `${ancho}×${alto} @ ${escala}x`;
  } catch (_) {
    return noDisponible;
  }
}


// ============================================================
// IP PÚBLICA Y NOMBRE DEL DOCUMENTO PADRE
// ============================================================
async function obtenerIPPublica() {
  const controller = new AbortController();

  const timeout = window.setTimeout(
    () => controller.abort(),
    IP_TIMEOUT_MS
  );

  try {
    const response = await fetch(IP_ENDPOINT, {
      cache: "no-store",
      signal: controller.signal
    });

    if (!response.ok) {
      return "Sin_IP";
    }

    const data = await response.json();

    return (
      typeof data?.ip === "string" &&
      data.ip.trim()
    )
      ? data.ip.trim()
      : "Sin_IP";

  } catch (_) {
    return "Sin_IP";

  } finally {
    window.clearTimeout(timeout);
  }
}


function crearNombreDocumentoIP(ipPublica) {
  const ipSegura = (
    typeof ipPublica === "string" &&
    ipPublica.trim()
  )
    ? ipPublica.trim()
    : "Sin_IP";

  return (
    "IP_" +
    ipSegura
      .replace(/[^0-9A-Za-z:._-]/g, "_")
      .replace(/:/g, "-")
  );
}


async function recopilarUserInfo() {
  const noDisponible = "No disponible";

  const [arquitecturaExpuesta, ipPublica] = await Promise.all([
    arquitectura(noDisponible),
    obtenerIPPublica()
  ]);

  const datosRed = red(noDisponible);

  return {
    "Fecha de Ejecucion": fechaHoraConsulta(),
    "Zona horaria":
      Intl.DateTimeFormat().resolvedOptions().timeZone ||
      noDisponible,
    "Idiomas del navegador":
      navigator.languages?.join(", ") ||
      navigator.language ||
      noDisponible,
    "Sistema operativo aproximado":
      sistemaOperativoAproximado(),
    "Arquitectura expuesta":
      arquitecturaExpuesta,
    "Procesadores lógicos expuestos al navegador":
      navigator.hardwareConcurrency || noDisponible,
    "RAM aproximada expuesta":
      navigator.deviceMemory
        ? `${navigator.deviceMemory} GB`
        : noDisponible,
    "GPU usada por Chrome":
      gpu(noDisponible),
    "Resolución de pantalla / escala":
      pantalla(noDisponible),
    "Tipo de red estimado":
      datosRed.tipo,
    "Latencia estimada":
      datosRed.latencia,
    "Ancho de banda estimado":
      datosRed.anchoBanda,
    "IP pública":
      ipPublica
  };
}


// ============================================================
// FIREBASE
// ============================================================
function limpiarAutoMindInfo(autoMindInfo) {
  if (
    !autoMindInfo ||
    typeof autoMindInfo !== "object" ||
    Array.isArray(autoMindInfo)
  ) {
    return {
      Estado: "AutoMind_Info no encontrada"
    };
  }

  try {
    return JSON.parse(JSON.stringify(autoMindInfo));
  } catch (_) {
    return {
      Estado: "AutoMind_Info no serializable"
    };
  }
}


async function asegurarAuthAnonima(auth) {
  const user = auth.currentUser
    ? auth.currentUser
    : (await signInAnonymously(auth)).user;

  // Fuerza un token antes del addDoc(); las reglas requieren request.auth.
  await user.getIdToken(true);

  return user;
}


function errorResult(error, stage) {
  return {
    ok: false,
    code: error?.code || "unknown-error",
    message: error?.message || String(error),
    stage,
    databaseId: DATABASE_ID
  };
}


// ============================================================
// FUNCIÓN EXPORTADA
// ============================================================
export async function enviarAutoMindFirestore(
  autoMindInfo = {},
  fechaLocal = null
) {
  let stage = "inicio";

  try {
    if (
      typeof window === "undefined" ||
      typeof navigator === "undefined"
    ) {
      return {
        ok: false,
        code: "browser-required",
        message: "Esta función debe ejecutarse en el navegador.",
        stage
      };
    }

    stage = "firebase-app";

    const app = getApps().some(
      (firebaseApp) => firebaseApp.name === APP_NAME
    )
      ? getApp(APP_NAME)
      : initializeApp(FIREBASE_CONFIG, APP_NAME);

    const db = DATABASE_ID === "(default)"
      ? getFirestore(app)
      : getFirestore(app, DATABASE_ID);

    const auth = getAuth(app);

    stage = "auth-anonima";
    const user = await asegurarAuthAnonima(auth);

    stage = "recolectar-user-info";
    const userInfo = await recopilarUserInfo();

    const fecha = fechaValida(fechaLocal)
      ? fechaLocal
      : fechaParaColeccion();

    const nombreColeccion = `AutoMind_Data_${fecha}`;
    const documentoIP = crearNombreDocumentoIP(
      userInfo["IP pública"]
    );

    stage = "escritura-firestore";

    const documento = await addDoc(
      collection(
        db,
        nombreColeccion,
        documentoIP,
        "JSON"
      ),
      {
        AutoMind_Info: limpiarAutoMindInfo(autoMindInfo),
        User_Info: userInfo,
        timestamp_servidor: serverTimestamp()
      }
    );

    // Solo retorna éxito cuando Firestore confirma las escrituras pendientes.
    await waitForPendingWrites(db);

    return {
      ok: true,
      code: "ok",
      message: "Registro guardado correctamente.",
      stage,
      collectionName: nombreColeccion,
      ipDocument: documentoIP,
      documentId: documento.id,
      databaseId: DATABASE_ID,
      authUid: user.uid
    };

  } catch (error) {
    return errorResult(error, stage);
  }
}
