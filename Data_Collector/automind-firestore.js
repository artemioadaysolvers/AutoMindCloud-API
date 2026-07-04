// automind-firestore.js
// Envía AutoMind_Info + información del navegador a Firestore.
// No imprime nada en consola ni muestra mensajes visuales.

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


// ============================================================
// CONFIGURACIÓN FIJA DEL PROYECTO
// ============================================================
const FIREBASE_CONFIG = {
  apiKey: "AIzaSyBSC-OGbSo_8wJlv9nSLJ8lUojcEKimOBQ",
  authDomain: "automindrobotics.firebaseapp.com",
  projectId: "automindrobotics",
  storageBucket: "automindrobotics.firebasestorage.app",
  messagingSenderId: "619255898589",
  appId: "1:619255898589:web:24605a66f71f9f9ae71634"
};

const DATABASE_ID = "automindcolab";
const COLLECTION_NAME = "automind_data";
const APP_NAME = "automind-firestore-app";

const CONSULTAR_IP_PUBLICA = true;
const INCLUIR_GPU = true;
const INCLUIR_RED = true;


// ============================================================
// INFORMACIÓN DEL NAVEGADOR
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


async function getArchitecture(unavailable) {
  try {
    if (navigator.userAgentData?.getHighEntropyValues) {
      const data = await navigator.userAgentData.getHighEntropyValues([
        "architecture",
        "bitness"
      ]);

      return [
        data.architecture,
        data.bitness ? `${data.bitness}-bit` : null
      ]
        .filter(Boolean)
        .join(" ") || unavailable;
    }

    const ua = navigator.userAgent || "";

    if (/Win64|x86_64|x64/i.test(ua)) return "x86_64";
    if (/ARM64|aarch64/i.test(ua)) return "ARM64";
    if (/x86/i.test(ua)) return "x86";

    return unavailable;

  } catch {
    return unavailable;
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

    return `${vendor} | ${renderer}`;

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


async function getPublicIP(unavailable) {
  if (!CONSULTAR_IP_PUBLICA) {
    return "No consultada";
  }

  try {
    const response = await fetch(
      "https://api.ipify.org?format=json",
      { cache: "no-store" }
    );

    if (!response.ok) {
      return unavailable;
    }

    const data = await response.json();

    return data.ip || unavailable;

  } catch {
    return unavailable;
  }
}


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
      `${screen.width}×${screen.height} @ ${window.devicePixelRatio}x`,

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
// Solo recibe AutoMind_Info y envía todo silenciosamente.
// ============================================================
export async function enviarAutoMindFirestore(autoMindInfo = {}) {
  try {
    const firebaseAppExiste = getApps().some(
      (app) => app.name === APP_NAME
    );

    const app = firebaseAppExiste
      ? getApp(APP_NAME)
      : initializeApp(FIREBASE_CONFIG, APP_NAME);

    const db = getFirestore(app, DATABASE_ID);

    const userInfo = await recopilarUserInfo();

    const autoMindInfoSeguro =
      autoMindInfo &&
      typeof autoMindInfo === "object" &&
      !Array.isArray(autoMindInfo)
        ? autoMindInfo
        : {
            Estado: "AutoMind_Info no encontrada"
          };

    await addDoc(
      collection(db, COLLECTION_NAME),
      {
        AutoMind_Info: autoMindInfoSeguro,
        User_Info: userInfo,
        timestamp_servidor: serverTimestamp()
      }
    );

    return true;

  } catch {
    return false;
  }
}
