// automind-firestore.js
// Envia AutoMind_Info + User_Info a Firestore.
// Guarda en: AutoMind_Data_DD-MM-AAAA / IP_xxx / ID_xxx
// No escribe en consola ni muestra mensajes visuales.

import {
  initializeApp,
  getApps,
  getApp
} from "https://www.gstatic.com/firebasejs/9.22.0/firebase-app.js";

import {
  getAuth,
  signInAnonymously
} from "https://www.gstatic.com/firebasejs/9.22.0/firebase-auth.js";

import {
  getFirestore,
  doc,
  setDoc,
  serverTimestamp
} from "https://www.gstatic.com/firebasejs/9.22.0/firebase-firestore.js";


// ============================================================
// CONFIGURACION FIJA DEL PROYECTO
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
const MODULE_VERSION = "automind-firestore-fecha-ip-id-2026-07-04-01";

const DATE_COLLECTION_PREFIX = "AutoMind_Data_";

const CONSULTAR_IP_PUBLICA = true;
const INCLUIR_GPU = true;
const INCLUIR_RED = true;

const IP_ENDPOINT = "https://api.ipify.org?format=json";
const IP_TIMEOUT_MS = 8000;
const AUTH_TIMEOUT_MS = 8000;
const FIRESTORE_TIMEOUT_MS = 12000;


// ============================================================
// UTILIDADES DE FECHA Y RUTA FIRESTORE
// ============================================================
function getPartesFecha(ahora = new Date()) {
  const dd = String(ahora.getDate()).padStart(2, "0");
  const mm = String(ahora.getMonth() + 1).padStart(2, "0");
  const yyyy = String(ahora.getFullYear());

  return {
    dd,
    mm,
    yyyy
  };
}


function getFechaColeccion(ahora = new Date()) {
  const { dd, mm, yyyy } = getPartesFecha(ahora);
  return `${DATE_COLLECTION_PREFIX}${dd}-${mm}-${yyyy}`;
}


function getFechaHoraConsulta(ahora = new Date()) {
  const { dd, mm, yyyy } = getPartesFecha(ahora);

  const hh = String(ahora.getHours()).padStart(2, "0");
  const min = String(ahora.getMinutes()).padStart(2, "0");
  const ss = String(ahora.getSeconds()).padStart(2, "0");

  return `${dd}-${mm}-${yyyy}-${hh}:${min}:${ss}`;
}


function getRegistroId(ahora = new Date()) {
  const { dd, mm, yyyy } = getPartesFecha(ahora);
  const hh = String(ahora.getHours()).padStart(2, "0");
  const min = String(ahora.getMinutes()).padStart(2, "0");
  const ss = String(ahora.getSeconds()).padStart(2, "0");

  const random =
    crypto?.randomUUID?.().replace(/-/g, "").slice(0, 12) ||
    Math.random().toString(36).slice(2, 14);

  return `ID_${dd}-${mm}-${yyyy}_${hh}-${min}-${ss}_${random}`;
}


function limpiarSegmentoRuta(valor, fallback) {
  const limpio = String(valor || "")
    .trim()
    .replace(/[^0-9A-Za-z:._-]/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "");

  return limpio || fallback;
}


function getDocumentoIP(ipPublica) {
  return `IP_${limpiarSegmentoRuta(ipPublica, "No_disponible")}`;
}


function conTimeout(promesa, tiempoMs, code) {
  let timeoutId;

  const timeout = new Promise((_, reject) => {
    timeoutId = setTimeout(() => {
      reject(
        Object.assign(
          new Error(`Timeout despues de ${tiempoMs} ms`),
          { code }
        )
      );
    }, tiempoMs);
  });

  return Promise.race([promesa, timeout]).finally(() => {
    clearTimeout(timeoutId);
  });
}


// ============================================================
// UTILIDADES DE NAVEGADOR
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

      const architecture = String(data.architecture || "").trim();
      const bitness = String(data.bitness || "").trim();

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

    return `${vendor || "Sin vendor"} | ${renderer || "Sin renderer"}`;

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

    return `${width}x${height} @ ${scale}x`;

  } catch {
    return unavailable;
  }
}


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

    return data?.ip || unavailable;

  } catch {
    return unavailable;

  } finally {
    clearTimeout(timeoutId);
  }
}


async function asegurarAutenticacion(app) {
  const auth = getAuth(app);

  if (auth.currentUser) {
    return auth.currentUser;
  }

  const credencial = await signInAnonymously(auth);
  return credencial.user;
}


// ============================================================
// RECOLECCION DE User_Info
// ============================================================
async function recopilarUserInfo(ahora = new Date()) {
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
    "Fecha de Ejecucion": getFechaHoraConsulta(ahora),

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

    "Procesadores logicos expuestos al navegador":
      navigator.hardwareConcurrency || unavailable,

    "RAM aproximada expuesta":
      navigator.deviceMemory
        ? `${navigator.deviceMemory} GB`
        : unavailable,

    "GPU usada por Chrome":
      INCLUIR_GPU
        ? getGPU(unavailable)
        : "No consultada",

    "Resolucion de pantalla / escala":
      getScreenInfo(unavailable),

    "Tipo de red estimado":
      network.type,

    "Latencia estimada":
      network.rtt,

    "Ancho de banda estimado":
      network.downlink,

    "IP publica":
      publicIP
  };
}


// ============================================================
// FUNCION PUBLICA
// Solo recibe AutoMind_Info.
// Devuelve resultado, pero no imprime nada.
// ============================================================
export async function enviarAutoMindFirestore(autoMindInfo = {}) {
  try {
    if (
      typeof window === "undefined" ||
      typeof navigator === "undefined"
    ) {
      return {
        ok: false,
        code: "browser-required",
        message: "Esta funcion debe ejecutarse desde un navegador."
      };
    }

    const firebaseAppExiste = getApps().some(
      (app) => app.name === APP_NAME
    );

    const app = firebaseAppExiste
      ? getApp(APP_NAME)
      : initializeApp(FIREBASE_CONFIG, APP_NAME);

    await conTimeout(
      asegurarAutenticacion(app),
      AUTH_TIMEOUT_MS,
      "auth-timeout"
    );

    const db = getFirestore(app, DATABASE_ID);
    const ahora = new Date();

    const userInfo = await recopilarUserInfo(ahora);

    const fechaColeccion = getFechaColeccion(ahora);
    const documentoIP = getDocumentoIP(userInfo["IP publica"]);

    const autoMindInfoSeguro =
      autoMindInfo &&
      typeof autoMindInfo === "object" &&
      !Array.isArray(autoMindInfo)
        ? autoMindInfo
        : {
            Estado: "AutoMind_Info no encontrada"
          };

    const registroId = getRegistroId(ahora);

    const referenciaIP = doc(
      db,
      fechaColeccion,
      documentoIP
    );

    await conTimeout(
      setDoc(
        referenciaIP,
        {
          [registroId]: {
            AutoMind_Info: autoMindInfoSeguro,
            User_Info: userInfo,
            timestamp_servidor: serverTimestamp()
          }
        },
        {
          merge: true
        }
      ),
      FIRESTORE_TIMEOUT_MS,
      "firestore-timeout"
    );

    return {
      ok: true,
      version: MODULE_VERSION,
      documentId: documentoIP,
      registroId,
      path: `${fechaColeccion}/${documentoIP}/${registroId}`
    };

  } catch (error) {
    return {
      ok: false,
      version: MODULE_VERSION,
      code: error?.code || "unknown-error",
      message: error?.message || String(error)
    };
  }
}
