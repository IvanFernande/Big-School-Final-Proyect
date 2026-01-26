TEST_SET = [
    # ======================
    # KPIs (CSV)
    # ======================
    {
        "question": "Cual es el SLA cumplido en Q4 y cuantos tickets abiertos hubo?",
        "expected": ["93.1", "1500"],
        "expect_text": "En Q4 el SLA cumplido es 93.1% y hubo 1500 tickets abiertos.",
        "category": "tablas",
    },
    {
        "question": "Cual es el CSAT promedio en Q3?",
        "expected": ["4.4"],
        "expect_text": "El CSAT promedio en Q3 es 4.4.",
        "category": "tablas",
    },
    {
        "question": "Cual es el SLA cumplido en Q2 y cuantos tickets cerrados hubo?",
        "expected": ["92.7", "1290"],
        "expect_text": "En Q2 el SLA cumplido es 92.7% y hubo 1290 tickets cerrados.",
        "category": "tablas",
    },
    {
        "question": "Que trimestre tuvo el SLA mas bajo?",
        "expected": ["q3", "89.9"],
        "expect_text": "El trimestre con el SLA más bajo fue Q3 con 89.9%.",
        "category": "tablas",
    },
    {
        "question": "Ordena los trimestres por tickets abiertos de mayor a menor",
        "expected": ["q4", "q3", "q2", "q1"],
        "expect_text": "Orden de tickets abiertos (mayor a menor): Q4, Q3, Q2, Q1.",
        "category": "tablas",
    },

    # ======================
    # Configuración servicio (JSON)
    # ======================
    {
        "question": "Cual es el servicio principal configurado?",
        "expected": ["core_api"],
        "expect_text": "El servicio principal configurado es core_api.",
        "category": "configuracion",
    },
    {
        "question": "Cual es la version del servicio principal?",
        "expected": ["2.3.1"],
        "expect_text": "La versión del servicio principal (core_api) es 2.3.1.",
        "category": "identificadores",
    },
    {
        "question": "Que features principales incluye el servicio core_api?",
        "expected": ["autenticacion_segura", "monitorizacion_activa", "autoscaling"],
        "expect_text": "Las features principales de core_api son: autenticacion_segura, monitorizacion_activa y autoscaling.",
        "category": "configuracion",
    },
    {
        "question": "Cuales son los umbrales de alerting cpu, memoria y latencia?",
        "expected": ["85", "80", "250"],
        "expect_text": "Umbrales de alerting: CPU 85%, memoria 80% y latencia 250 ms.",
        "category": "configuracion",
    },
    {
        "question": "Quien es la responsable de soporte y su email?",
        "expected": ["laura", "gomez", "@technova.com"],
        "expect_text": "La responsable de soporte es Laura Gomez y su email es laura.gomez@technova.com.",
        "category": "identificadores",
    },

    # ======================
    # FAQ
    # ======================
    {
        "question": "Como restablecer la contrasena?",
        "expected": ["restablecer", "portal", "ticket"],
        "expect_text": "Para restablecer la contraseña: usa el portal de autoservicio; si no funciona, abre un ticket a soporte.",
        "category": "procedimientos",
    },
    {
        "question": "Que hacer si no puedo acceder a la vpn?",
        "expected": ["vpn", "ticket"],
        "expect_text": "Si no puedes acceder a la VPN: revisa tu conexión y credenciales; si persiste, abre un ticket indicando el error.",
        "category": "procedimientos",
    },
    {
        "question": "Que datos debo incluir al reportar una incidencia?",
        "expected": ["capturas", "descripcion", "pasos"],
        "expect_text": "Incluye al reportar una incidencia: descripción del problema, pasos para reproducirlo y capturas/evidencias.",
        "category": "procedimientos",
    },
    {
        "question": "Cuales son los tiempos normales de respuesta?",
        "expected": ["15", "2 horas"],
        "expect_text": "Tiempos normales de respuesta: 15 minutos para incidencias críticas y 2 horas para el resto según prioridad.",
        "category": "procedimientos",
    },

    # ======================
    # Manual + README
    # ======================
    {
        "question": "Resume el flujo de escalado del sistema.",
        "expected": ["escalado", "sla", "nivel 1"],
        "expect_text": "Flujo de escalado: Nivel 1 triage y diagnóstico inicial; si impacta SLA o no se resuelve, se escala a Nivel 2/3 con contexto y evidencias.",
        "category": "procedimientos",
    },
    {
        "question": "Que modulos componen el sistema de soporte?",
        "expected": ["tickets", "monitorizacion", "diagnóstico", "kpi"],
        "expect_text": "El sistema de soporte se compone de módulos de tickets, monitorización, diagnóstico y KPIs.",
        "category": "procedimientos",
    },
    {
        "question": "Que pasos de troubleshooting se realizan antes de escalar?",
        "expected": ["verificar", "credenciales", "logs"],
        "expect_text": "Antes de escalar se realiza troubleshooting: verificar credenciales, revisar logs/errores y comprobar estado del servicio y dependencias.",
        "category": "procedimientos",
    },

    # ======================
    # On-call
    # ======================
    {
        "question": "Cual es el horario de guardia entre semana y fin de semana?",
        "expected": ["18:00", "08:00", "24h"],
        "expect_text": "Horario on-call: entre semana de 18:00 a 08:00; fin de semana 24h.",
        "category": "procedimientos",
    },
    {
        "question": "Que tiempo maximo tiene el tecnico oncall para responder una alerta?",
        "expected": ["10 minutos"],
        "expect_text": "El técnico on-call tiene un máximo de 10 minutos para responder una alerta.",
        "category": "procedimientos",
    },
    {
        "question": "Que hacer ante una alerta de servicio degradado?",
        "expected": ["latencia", "reiniciar", "estado"],
        "expect_text": "Ante servicio degradado: validar el estado, revisar latencia/errores y, si procede, reiniciar el servicio siguiendo el runbook y comunicar el estado.",
        "category": "procedimientos",
    },
    {
        "question": "Que comandos se usan en oncall?",
        "expected": ["tn-monitor", "tn-restart", "tn-alert"],
        "expect_text": "Comandos on-call: tn-monitor, tn-restart y tn-alert.",
        "category": "procedimientos",
    },

    # ======================
    # Políticas operativas
    # ======================
    {
        "question": "Que canales oficiales deben usarse para comunicacion interna?",
        "expected": ["tickets", "correo"],
        "expect_text": "Canales oficiales para comunicación interna: sistema de tickets y correo corporativo.",
        "category": "procedimientos",
    },
    {
        "question": "Que requiere la politica de gestion de cambios antes de produccion?",
        "expected": ["aprobado", "evaluado"],
        "expect_text": "Antes de producción, la gestión de cambios requiere evaluación del riesgo/impacto y aprobación formal.",
        "category": "procedimientos",
    },
    {
        "question": "Como se priorizan incidencias segun impacto y urgencia?",
        "expected": ["prioridad", "impacto", "urgencia"],
        "expect_text": "Las incidencias se priorizan combinando impacto y urgencia para asignar una prioridad (alta, media o baja).",
        "category": "procedimientos",
    },

    # ======================
    # Inventarios
    # ======================
    {
        "question": "Cual es el estado del firewall principal y del ups?",
        "expected": ["firewall", "operativo", "ups", "mantenimiento"],
        "expect_text": "Estado inventario: el firewall principal está operativo y el UPS está en mantenimiento.",
        "category": "inventarios",
    },
    {
        "question": "Que servicio tiene criticidad alta y owner backend_team?",
        "expected": ["core_api", "alta", "backend_team"],
        "expect_text": "El servicio con criticidad alta y owner backend_team es core_api.",
        "category": "inventarios",
    },
    {
        "question": "Que servicio tiene criticidad media y owner infra_team?",
        "expected": ["notificaciones", "media", "infra_team"],
        "expect_text": "El servicio con criticidad media y owner infra_team es notificaciones.",
        "category": "inventarios",
    },

    # ======================
    # Histórico de incidentes
    # ======================
    {
        "question": "Cual fue la causa raiz del incidente INC-2024-001?",
        "expected": ["pool de conexiones"],
        "expect_text": "La causa raíz del incidente INC-2024-001 fue un problema en el pool de conexiones.",
        "category": "identificadores",
    },
    {
        "question": "Que acciones correctivas se tomaron en INC-2024-009?",
        "expected": ["rollback", "automatico"],
        "expect_text": "En el incidente INC-2024-009 se realizó un rollback automático como acción correctiva.",
        "category": "identificadores",
    },
    {
        "question": "Que incidente tuvo perdida de nodo y cuanto duro la resolucion?",
        "expected": ["srv-db-002", "4h 22m"],
        "expect_text": "El incidente con pérdida de nodo fue srv-db-002 y la resolución duró 4h 22m.",
        "category": "identificadores",
    },

    # ======================
    # Organigrama
    # ======================
    {
        "question": "Quien es el Ingeniero Nivel 2 y cual es su email?",
        "expected": ["carlos", "ruiz", "@technova.com"],
        "expect_text": "El Ingeniero de Nivel 2 es Carlos Ruiz y su email es carlos.ruiz@technova.com.",
        "category": "identificadores",
    },

    # ======================
    # Incidente crítico
    # ======================
    {
        "question": "Cuales son los pasos iniciales ante una incidencia critica?",
        "expected": ["clasificar", "impacto", "urgencia"],
        "expect_text": "Pasos iniciales ante incidencia crítica: clasificar el incidente, evaluar impacto y urgencia, y activar el protocolo de comunicación/escalado.",
        "category": "procedimientos",
    },
    {
        "question": "Que informacion minima debe registrarse en un ticket critico?",
        "expected": ["descripcion", "impacto", "urgencia", "evidencias"],
        "expect_text": "En un ticket crítico se debe registrar como mínimo: descripción, impacto, urgencia y evidencias (logs/capturas).",
        "category": "procedimientos",
    },
]
