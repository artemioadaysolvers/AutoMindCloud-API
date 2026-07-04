// automind-firestore.js
// Envía AutoMind_Info + User_Info a Firestore.
// No muestra mensajes visuales ni escribe en consola.
//
// Estructura:
// AutoMind_Data_DD-MM-AAAA
// └── IP_xxx.xxx.xxx.xxx
//     └── JSON
//         └── documento automático
//
// El documento IP_... no recibe campos.
// Solo actúa como padre visual de la subcolección JSON.

import {
  initializeApp,
  getApps,
  getApp
} from "https://www.gstatic.com/firebasejs/9.22.0/firebase-app.js";

import {
  getFirestore,
  collection,
  addDoc,
  serverTimestamp
} from "https://www.gstatic.com/firebasejs/9.22.0/firebase-firestore.js";

import {
  getAuth,
  signInAnonymously
} from "https://www.gstatic.com/firebasejs/9.22.0/firebase-auth.js";


// ============================================================
// CONFIGURACIÓN FIJA DEL PROYECTO
// ============================================================
const FIREBASE_CONFIG = Object.freeze({
  apiKey: "AIzaSyBSC-OGbSo_8wJlv9nSLJ8lUojcEKimOBQ",
  authDomain: "automindrobotics.firebaseapp.com",
  projectId: "automindrobotics",
  storageBucket: "automindrobotics.firebasestorage.app",
  messagingSenderId: "619255898589",
  appId: "1:619255898589:web:24605a66f71f9f9ae71634"
});

const DATABASE_ID = "automindcolab";
const APP_NAME = "automind-firestore-app";

const CONSULTAR_IP_PUBLICA = true;
const INCLUIR_GPU = true;
const INCLUIR_RED = true;

const IP_ENDPOINT = "https://api.ipify.org?format=json";
const IP_TIMEOUT_MS = 8000;


// ============================================================
// UTILIDADES DE FECHA
// ============================================================
function getFechaHoraConsulta() {
  const ahora = new Date();

  const dd = String(ahora.getDate()).padStart(2, "0");
  const mm = String(ahora.getMonth() + 1).padStart(2, "0");
  const yyyy = ahora.getFullYear();

  const hh = String(ahora.getHours()).padStart(2, "0");
  const min = String(ahora.getMinutes()).padStart(2, "0");
  const ss = String(ahora.getSeconds()).padStart(2, "0");

  return `${dd}-${mm}-${yyyy}-${hh}:${min}:${ss}`;
}


function getFechaParaColeccion() {
  const ahora = new Date();

  const dd = String(ahora.getDate()).padStart(2, "0");
  const mm = String(ahora.getMonth() + 1).padStart(2, "0");
  const yyyy = ahora.getFullYear();

  return `${dd}-${mm}-${yyyy}`;
}


function fechaEsValida(fecha) {
  return (
    typeof fecha === "string" &&
    /^\d{2}-\d{2}-\d{4}$/.test(fecha)
  );
}


function resolverFechaColeccion(fechaLocal) {
  if (fechaEsValida(fechaLocal)) {
    return fechaLocal;
  }

  return getFechaParaColeccion();
}


// ============================================================
// UTILIDADES DEL NAVEGADOR
// ============================================================
function getOSApprox() {
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


function getArchitectureFromUserAgent(unavailable) {
  const ua = navigator.userAgent || "";

  if (/ARM64|aarch64/i.test(ua)) return "ARM64";
  if (/Win64|x86_64|x64|amd64/i.test(ua)) return "x86_64";
  if (/i[3-6]86|x86/i.test(ua)) return "x86";

  return unavailable;
}


async function getArchitecture(unavailable) {
  try {
    if (navigator.userAgentData?.getHighEntropyValues) {
      const data = await navigator.userAgentData.getHighEntropyValues([
        "architecture",
        "bitness"
      ]);

      const architecture = String(
        data.architecture || ""
      ).trim();

      const bitness = String(
        data.bitness || ""
      ).trim();

      const resultado = [
        architecture,
        bitness ? `${bitness}-bit` : null
      ]
        .filter(Boolean)
        .join(" ");

      if (resultado) {
        return resultado;
      }
    }

    return getArchitectureFromUserAgent(unavailable);

  } catch {
    return getArchitectureFromUserAgent(unavailable);
  }
}


function getGPU(unavailable) {
  try {
    const canvas = document.createElement("canvas");

    const gl =
      canvas.getContext("webgl") ||
      canvas.getContext("experimental-webgl");

    if (!gl) {
      return unavailable;
    }

    const debugInfo = gl.getExtension(
      "WEBGL_debug_renderer_info"
    );

    if (!debugInfo) {
      return unavailable;
    }

    const vendor = gl.getParameter(
      debugInfo.UNMASKED_VENDOR_WEBGL
    );

    const renderer = gl.getParameter(
      debugInfo.UNMASKED_RENDERER_WEBGL
    );

    if (!vendor && !renderer) {
      return unavailable;
    }

    return (
      `${vendor || "Sin vendor"} | ` +
      `${renderer || "Sin renderer"}`
    );

  } catch {
    return unavailable;
  }
}


function getNetwork(unavailable) {
  const connection =
    navigator.connection ||
    navigator.mozConnection ||
    navigator.webkitConnection;

  if (!connection) {
    return {
      type: unavailable,
      rtt: unavailable,
      downlink: unavailable
    };
  }

  return {
    type: connection.effectiveType || unavailable,

    rtt:
      typeof connection.rtt === "number"
        ? `${connection.rtt} ms`
        : unavailable,

    downlink:
      typeof connection.downlink === "number"
        ? `${connection.downlink} Mbps`
        : unavailable
  };
}


function getScreenInfo(unavailable) {
  try {
    if (typeof screen === "undefined") {
      return unavailable;
    }

    const width = screen.width || unavailable;
    const height = screen.height || unavailable;
    const scale = window.devicePixelRatio || 1;

    return `${width}×${height} @ ${scale}x`;

  } catch {
    return unavailable;
  }
}


// ============================================================
// IP PÚBLICA Y NOMBRE DEL DOCUMENTO IP
// ============================================================
async function getPublicIP(unavailable) {
  if (!CONSULTAR_IP_PUBLICA) {
    return "No consultada";
  }

  const controller = new AbortController();

  const timeoutId = setTimeout(() => {
    controller.abort();
  }, IP_TIMEOUT_MS);

  try {
    const response = await fetch(IP_ENDPOINT, {
      cache: "no-store",
      signal: controller.signal
    });

    if (!response.ok) {
      return unavailable;
    }

    const data = await response.json();

    if (
      data &&
      typeof data.ip === "string" &&
      data.ip.trim()
    ) {
      return data.ip.trim();
    }

    return unavailable;

  } catch {
    return unavailable;

  } finally {
    clearTimeout(timeoutId);
  }
}


function getDocumentoIP(ipPublica) {
  const ipSegura = (
    typeof ipPublica === "string" &&
    ipPublica.trim()
      ? ipPublica.trim()
      : "Sin_IP"
  )
    .replace(/[^0-9A-Fa-f:.\-]/g, "_")
    .replace(/:/g, "-")
    .replace(/\//g, "_");

  return `IP_${ipSegura}`;
}


// ============================================================
// FIREBASE AUTH ANÓNIMO
// ============================================================
async function asegurarAuthAnonimo(auth) {
  if (auth.currentUser) {
    return auth.currentUser;
  }

  const credencial = await signInAnonymously(auth);

  return credencial.user;
}


// ============================================================
// RECOLECCIÓN DE User_Info
// ============================================================
async function recopilarUserInfo() {
  const unavailable = "No disponible";

  const [architecture, publicIP] = await Promise.all([
    getArchitecture(unavailable),
    getPublicIP(unavailable)
  ]);

  const network = INCLUIR_RED
    ? getNetwork(unavailable)
    : {
        type: "No consultada",
        rtt: "No consultada",
        downlink: "No consultada"
      };

  return {
    "Fecha de Ejecucion": getFechaHoraConsulta(),

    "Zona horaria":
      Intl.DateTimeFormat()
        .resolvedOptions()
        .timeZone || unavailable,

    "Idiomas del navegador":
      navigator.languages?.join(", ") ||
      navigator.language ||
      unavailable,

    "Sistema operativo aproximado":
      getOSApprox(),

    "Arquitectura expuesta":
      architecture,

    "Procesadores lógicos expuestos al navegador":
      navigator.hardwareConcurrency || unavailable,

    "RAM aproximada expuesta":
      navigator.deviceMemory
        ? `${navigator.deviceMemory} GB`
        : unavailable,

    "GPU usada por Chrome":
      INCLUIR_GPU
        ? getGPU(unavailable)
        : "No consultada",

    "Resolución de pantalla / escala":
      getScreenInfo(unavailable),

    "Tipo de red estimado":
      network.type,

    "Latencia estimada":
      network.rtt,

    "Ancho de banda estimado":
      network.downlink,

    "IP pública":
      publicIP
  };
}


// ============================================================
// FUNCIÓN PÚBLICA
//
// Recibe:
//   enviarAutoMindFirestore(autoMindInfo, fechaLocal)
//
// Ejemplo:
//   enviarAutoMindFirestore(info, "03-07-2026")
//
// Si fechaLocal no llega o no es válida, usa la fecha del navegador.
// ============================================================
export async function enviarAutoMindFirestore(
  autoMindInfo = {},
  fechaLocal = null
) {
  try {
    if (
      typeof window === "undefined" ||
      typeof navigator === "undefined"
    ) {
      return {
        ok: false,
        code: "browser-required",
        message: "Esta función debe ejecutarse desde un navegador."
      };
    }

    const firebaseAppExiste = getApps().some(
      (app) => app.name === APP_NAME
    );

    const app = firebaseAppExiste
      ? getApp(APP_NAME)
      : initializeApp(FIREBASE_CONFIG, APP_NAME);

    const db = getFirestore(app, DATABASE_ID);
    const auth = getAuth(app);

    await asegurarAuthAnonimo(auth);

    const userInfo = await recopilarUserInfo();

    const autoMindInfoSeguro =
      autoMindInfo &&
      typeof autoMindInfo === "object" &&
      !Array.isArray(autoMindInfo)
        ? autoMindInfo
        : {
            Estado: "AutoMind_Info no encontrada"
          };

    const fecha = resolverFechaColeccion(fechaLocal);

    const nombreColeccion = (
      `AutoMind_Data_${fecha}`
    );

    const documentoIP = getDocumentoIP(
      userInfo["IP pública"]
    );

    /*
      No se usa setDoc() ni se guardan campos en documentoIP.

      Se escribe directamente en:

      AutoMind_Data_DD-MM-AAAA
      └── IP_xxx.xxx.xxx.xxx
          └── JSON
              └── documento automático
    */
    const documento = await addDoc(
      collection(
        db,
        nombreColeccion,
        documentoIP,
        "JSON"
      ),
      {
        AutoMind_Info: autoMindInfoSeguro,
        User_Info: userInfo,
        timestamp_servidor: serverTimestamp()
      }
    );

    return {
      ok: true,
      collectionName: nombreColeccion,
      ipDocument: documentoIP,
      documentId: documento.id
    };

  } catch (error) {
    return {
      ok: false,
      code: error?.code || "unknown-error",
      message: error?.message || String(error)
    };
  }
}
