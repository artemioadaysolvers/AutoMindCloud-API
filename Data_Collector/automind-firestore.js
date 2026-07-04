// automind-firestore.js
// Envía AutoMind_Info + User_Info desde el navegador a Cloud Firestore.
//
// Estructura creada:
// AutoMind_Data_DD-MM-AAAA
// └── IP_xxx.xxx.xxx.xxx
//     └── JSON
//         └── documento automático
//
// Este módulo NO oculta los errores: la función pública devuelve un objeto
// { ok, code, message, ... } para que AutoMindCloud/__init__.py los muestre.

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
// CONFIGURACIÓN DEL PROYECTO
// ============================================================
const FIREBASE_CONFIG = Object.freeze({
  apiKey: "AIzaSyBSC-OGbSo_8wJlv9nSLJ8lUojcEKimOBQ",
  authDomain: "automindrobotics.firebaseapp.com",
  projectId: "automindrobotics",
  storageBucket: "automindrobotics.firebasestorage.app",
  messagingSenderId: "619255898589",
  appId: "1:619255898589:web:24605a66f71f9f9ae71634"
});

// Debe coincidir EXACTAMENTE con el ID de Firestore creado en Google Cloud.
// Usa "(default)" solamente si esa es la base que creaste.
const DATABASE_ID = "automindcolab";
const APP_NAME = "automind-firestore-app";

const CONSULTAR_IP_PUBLICA = true;
const INCLUIR_GPU = true;
const INCLUIR_RED = true;

const IP_ENDPOINT = "https://api.ipify.org?format=json";
const IP_TIMEOUT_MS = 8000;

// Reintentos ante fallos transitorios de red/servicio. No reintenta errores
// permanentes como permission-denied o auth/operation-not-allowed.
const WRITE_RETRY_DELAYS_MS = Object.freeze([0, 900, 2200]);


// ============================================================
// UTILIDADES GENERALES
// ============================================================
function esperar(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}


function normalizarCodigoError(error) {
  const code = String(error?.code || "unknown-error").trim();
  return code || "unknown-error";
}


function codigoBase(error) {
  return normalizarCodigoError(error)
    .replace(/^firebase\//, "")
    .replace(/^firestore\//, "")
    .replace(/^auth\//, "");
}


function esErrorReintentable(error) {
  return new Set([
    "unavailable",
    "deadline-exceeded",
    "aborted",
    "resource-exhausted",
    "network-request-failed",
    "internal"
  ]).has(codigoBase(error));
}


function describirError(error, etapa) {
  const code = normalizarCodigoError(error);
  const base = codigoBase(error);
  const detalleOriginal = String(
    error?.message || error || "Error desconocido"
  );

  const prefijo = etapa ? `[${etapa}] ` : "";

  const mensajes = {
    "operation-not-allowed":
      "La autenticación anónima está deshabilitada. " +
      "Actívala en Firebase Authentication > Sign-in method > Anonymous.",

    "unauthorized-domain":
      "Firebase Auth rechazó el dominio de Colab. " +
      "Autoriza colab.research.google.com en Authentication > Settings > Authorized domains.",

    "network-request-failed":
      "No fue posible contactar Firebase Auth. Verifica la conexión y que el navegador no bloquee la solicitud.",

    "permission-denied":
      "Firestore rechazó la escritura. Revisa las reglas de la base " +
      `\"${DATABASE_ID}\" y confirma que permitan create cuando request.auth != null.`,

    "not-found":
      "No se encontró la base Firestore indicada. Confirma que DATABASE_ID sea " +
      `\"${DATABASE_ID}\" o usa \"(default)\" si corresponde.`,

    "failed-precondition":
      "Firestore no está listo para esta operación. Verifica que la base exista, sea Firestore Native/Standard y tenga reglas publicadas.",

    "invalid-argument":
      "Firestore rechazó algún dato enviado. AutoMind_Info debe contener únicamente valores compatibles con JSON.",

    "resource-exhausted":
      "Firestore rechazó temporalmente la operación por cuota o límite. El módulo intentó reintentarla.",

    "unavailable":
      "Firestore no está disponible temporalmente o la red impidió la conexión.",

    "deadline-exceeded":
      "La escritura superó el tiempo de espera de Firestore.",

    "offline":
      "El navegador no tiene conexión de red en este momento."
  };

  return {
    code,
    message: `${prefijo}${mensajes[base] || detalleOriginal}`,
    detalleOriginal
  };
}


async function ejecutarConReintentos(operacion) {
  let ultimoError = null;

  for (let intento = 0; intento < WRITE_RETRY_DELAYS_MS.length; intento += 1) {
    if (WRITE_RETRY_DELAYS_MS[intento] > 0) {
      await esperar(WRITE_RETRY_DELAYS_MS[intento]);
    }

    try {
      const valor = await operacion();
      return {
        valor,
        intentos: intento + 1
      };
    } catch (error) {
      ultimoError = error;

      const quedanIntentos = intento < WRITE_RETRY_DELAYS_MS.length - 1;

      if (!quedanIntentos || !esErrorReintentable(error)) {
        throw error;
      }
    }
  }

  throw ultimoError || new Error("No fue posible ejecutar la operación.");
}


// ============================================================
// FECHA
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
  return fechaEsValida(fechaLocal)
    ? fechaLocal
    : getFechaParaColeccion();
}


// ============================================================
// INFORMACIÓN EXPUESTA POR EL NAVEGADOR
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

      const result = [
        architecture,
        bitness ? `${bitness}-bit` : null
      ]
        .filter(Boolean)
        .join(" ");

      if (result) {
        return result;
      }
    }
  } catch (_) {
    // Se usa el fallback basado en userAgent más abajo.
  }

  return getArchitectureFromUserAgent(unavailable);
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

    const debugInfo = gl.getExtension("WEBGL_debug_renderer_info");

    if (!debugInfo) {
      return unavailable;
    }

    const vendor = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL);
    const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);

    if (!vendor && !renderer) {
      return unavailable;
    }

    return `${vendor || "Sin vendor"} | ${renderer || "Sin renderer"}`;
  } catch (_) {
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
  } catch (_) {
    return unavailable;
  }
}


// ============================================================
// IP PÚBLICA
// ============================================================
async function getPublicIP(unavailable) {
  if (!CONSULTAR_IP_PUBLICA) {
    return "No consultada";
  }

  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), IP_TIMEOUT_MS);

  try {
    const response = await fetch(IP_ENDPOINT, {
      cache: "no-store",
      signal: controller.signal
    });

    if (!response.ok) {
      return unavailable;
    }

    const data = await response.json();
    const ip = typeof data?.ip === "string" ? data.ip.trim() : "";

    return ip || unavailable;
  } catch (_) {
    // La ausencia de IP no impide guardar el registro.
    return unavailable;
  } finally {
    window.clearTimeout(timeoutId);
  }
}


function getDocumentoIP(ipPublica) {
  const ipSegura = (
    typeof ipPublica === "string" && ipPublica.trim()
      ? ipPublica.trim()
      : "Sin_IP"
  )
    .replace(/[^0-9A-Fa-f:.\-]/g, "_")
    .replace(/:/g, "-")
    .replace(/\//g, "_");

  return `IP_${ipSegura}`;
}


// ============================================================
// FIREBASE
// ============================================================
function obtenerFirebaseApp() {
  const existe = getApps().some((app) => app.name === APP_NAME);

  return existe
    ? getApp(APP_NAME)
    : initializeApp(FIREBASE_CONFIG, APP_NAME);
}


function obtenerFirestore(app) {
  // Para la base predeterminada evitamos pasar databaseId.
  // Para una base nombrada se usa su ID explícitamente.
  return DATABASE_ID === "(default)"
    ? getFirestore(app)
    : getFirestore(app, DATABASE_ID);
}


async function asegurarAuthAnonimo(auth) {
  const user = auth.currentUser || (await signInAnonymously(auth)).user;

  // Obliga a que Firebase termine de obtener un token antes del addDoc().
  // Así las reglas que exigen request.auth != null reciben la sesión ya lista.
  await user.getIdToken();

  return user;
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
      Intl.DateTimeFormat().resolvedOptions().timeZone || unavailable,
    "Idiomas del navegador":
      navigator.languages?.join(", ") ||
      navigator.language ||
      unavailable,
    "Sistema operativo aproximado": getOSApprox(),
    "Arquitectura expuesta": architecture,
    "Procesadores lógicos expuestos al navegador":
      navigator.hardwareConcurrency || unavailable,
    "RAM aproximada expuesta":
      navigator.deviceMemory
        ? `${navigator.deviceMemory} GB`
        : unavailable,
    "GPU usada por Chrome":
      INCLUIR_GPU ? getGPU(unavailable) : "No consultada",
    "Resolución de pantalla / escala": getScreenInfo(unavailable),
    "Tipo de red estimado": network.type,
    "Latencia estimada": network.rtt,
    "Ancho de banda estimado": network.downlink,
    "IP pública": publicIP
  };
}


function normalizarAutoMindInfo(autoMindInfo) {
  if (
    !autoMindInfo ||
    typeof autoMindInfo !== "object" ||
    Array.isArray(autoMindInfo)
  ) {
    return { Estado: "AutoMind_Info no encontrada" };
  }

  // Hace una copia JSON pura, previniendo undefined, funciones o valores
  // incompatibles con Firestore si la función se invoca manualmente.
  try {
    return JSON.parse(JSON.stringify(autoMindInfo));
  } catch (_) {
    return {
      Estado: "AutoMind_Info no serializable",
      Detalle: "El objeto debe contener datos compatibles con JSON."
    };
  }
}


// ============================================================
// FUNCIÓN PÚBLICA
// ============================================================
// enviarAutoMindFirestore(autoMindInfo, fechaLocal)
//
// Ejemplo:
// const r = await enviarAutoMindFirestore(info, "03-07-2026");
//
// Devuelve siempre un objeto. No lanza hacia el caller: AutoMindCloud
// puede mostrar el código y detalle exactos en la salida de Colab.
export async function enviarAutoMindFirestore(
  autoMindInfo = {},
  fechaLocal = null
) {
  let etapa = "inicio";

  try {
    if (
      typeof window === "undefined" ||
      typeof navigator === "undefined"
    ) {
      return {
        ok: false,
        code: "browser-required",
        message: "Esta función debe ejecutarse desde el navegador de Colab.",
        stage: etapa
      };
    }

    if (navigator.onLine === false) {
      return {
        ok: false,
        code: "offline",
        message: describirError({ code: "offline" }, "red").message,
        stage: "red"
      };
    }

    etapa = "firebase-app";
    const app = obtenerFirebaseApp();
    const db = obtenerFirestore(app);
    const auth = getAuth(app);

    etapa = "auth-anonima";
    const user = await asegurarAuthAnonimo(auth);

    etapa = "recolectar-user-info";
    const userInfo = await recopilarUserInfo();
    const autoMindInfoSeguro = normalizarAutoMindInfo(autoMindInfo);

    const fecha = resolverFechaColeccion(fechaLocal);
    const nombreColeccion = `AutoMind_Data_${fecha}`;
    const documentoIP = getDocumentoIP(userInfo["IP pública"]);

    etapa = "escritura-firestore";
    const resultado = await ejecutarConReintentos(() =>
      addDoc(
        collection(db, nombreColeccion, documentoIP, "JSON"),
        {
          AutoMind_Info: autoMindInfoSeguro,
          User_Info: userInfo,
          timestamp_servidor: serverTimestamp()
        }
      )
    );

    return {
      ok: true,
      code: "ok",
      message: "Registro guardado correctamente en Firestore.",
      stage: etapa,
      collectionName: nombreColeccion,
      ipDocument: documentoIP,
      documentId: resultado.valor.id,
      attempts: resultado.intentos,
      databaseId: DATABASE_ID,
      authUid: user.uid
    };
  } catch (error) {
    const descripcion = describirError(error, etapa);

    return {
      ok: false,
      code: descripcion.code,
      message: descripcion.message,
      originalMessage: descripcion.detalleOriginal,
      stage: etapa,
      databaseId: DATABASE_ID
    };
  }
}
