TEST_SET = [
    # ======================
    # KPIs (CSV)
    # ======================
    {"question": "Cual es el SLA cumplido en Q4 y cuantos tickets abiertos hubo?", "expected": ["93.1", "1500"], "category": "tablas"},
    {"question": "Cual es el CSAT promedio en Q3?", "expected": ["4.4"], "category": "tablas"},
    {"question": "Cual es el SLA cumplido en Q2 y cuantos tickets cerrados hubo?", "expected": ["92.7", "1290"], "category": "tablas"},
    {"question": "Que trimestre tuvo el SLA mas bajo?", "expected": ["q3", "89.9"], "category": "tablas"},
    {"question": "Ordena los trimestres por tickets abiertos de mayor a menor", "expected": ["q4", "q3", "q2", "q1"], "category": "tablas"},

    # ======================
    # Configuración servicio (JSON)
    # ======================
    {"question": "Cual es el servicio principal configurado?", "expected": ["core_api"], "category": "configuracion"},
    {"question": "Cual es la version del servicio principal?", "expected": ["2.3.1"], "category": "identificadores"},
    {"question": "Que features principales incluye el servicio core_api?", "expected": ["autenticacion_segura", "monitorizacion_activa", "autoscaling"], "category": "configuracion"},
    {"question": "Cuales son los umbrales de alerting cpu, memoria y latencia?", "expected": ["85", "80", "250"], "category": "configuracion"},
    {"question": "Quien es la responsable de soporte y su email?", "expected": ["laura", "gomez", "@technova.com"], "category": "identificadores"},

    # ======================
    # FAQ
    # ======================
    {"question": "Como restablecer la contrasena?", "expected": ["restablecer", "portal", "ticket"], "category": "procedimientos"},
    {"question": "Que hacer si no puedo acceder a la vpn?", "expected": ["vpn", "ticket"], "category": "procedimientos"},
    {"question": "Que datos debo incluir al reportar una incidencia?", "expected": ["capturas", "descripcion", "pasos"], "category": "procedimientos"},
    {"question": "Cuales son los tiempos normales de respuesta?", "expected": ["15", "2 horas"], "category": "procedimientos"},

    # ======================
    # Manual + README
    # ======================
    {"question": "Resume el flujo de escalado del sistema.", "expected": ["escalado", "sla", "nivel 1"], "category": "procedimientos"},
    {"question": "Que modulos componen el sistema de soporte?", "expected": ["tickets", "monitorizacion", "diagnóstico", "kpi"], "category": "procedimientos"},
    {"question": "Que pasos de troubleshooting se realizan antes de escalar?", "expected": ["verificar", "credenciales", "logs"], "category": "procedimientos"},

    # ======================
    # On-call
    # ======================
    {"question": "Cual es el horario de guardia entre semana y fin de semana?", "expected": ["18:00", "08:00", "24h"], "category": "procedimientos"},
    {"question": "Que tiempo maximo tiene el tecnico oncall para responder una alerta?", "expected": ["10 minutos"], "category": "procedimientos"},
    {"question": "Que hacer ante una alerta de servicio degradado?", "expected": ["latencia", "reiniciar", "estado"], "category": "procedimientos"},
    {"question": "Que comandos se usan en oncall?", "expected": ["tn-monitor", "tn-restart", "tn-alert"], "category": "procedimientos"},

    # ======================
    # Políticas operativas
    # ======================
    {"question": "Que canales oficiales deben usarse para comunicacion interna?", "expected": ["tickets", "correo"], "category": "procedimientos"},
    {"question": "Que requiere la politica de gestion de cambios antes de produccion?", "expected": ["aprobado", "evaluado"], "category": "procedimientos"},
    {"question": "Como se priorizan incidencias segun impacto y urgencia?", "expected": ["prioridad", "impacto", "urgencia"], "category": "procedimientos"},

    # ======================
    # Inventarios
    # ======================
    {"question": "Cual es el estado del firewall principal y del ups?", "expected": ["firewall", "operativo", "ups", "mantenimiento"], "category": "inventarios"},
    {"question": "Que servicio tiene criticidad alta y owner backend_team?", "expected": ["core_api", "alta", "backend_team"], "category": "inventarios"},
    {"question": "Que servicio tiene criticidad media y owner infra_team?", "expected": ["notificaciones", "media", "infra_team"], "category": "inventarios"},

    # ======================
    # Histórico de incidentes
    # ======================
    {"question": "Cual fue la causa raiz del incidente INC-2024-001?", "expected": ["pool de conexiones"], "category": "identificadores"},
    {"question": "Que acciones correctivas se tomaron en INC-2024-009?", "expected": ["rollback", "automatico"], "category": "identificadores"},
    {"question": "Que incidente tuvo perdida de nodo y cuanto duro la resolucion?", "expected": ["srv-db-002", "4h 22m"], "category": "identificadores"},

    # ======================
    # Organigrama
    # ======================
    {"question": "Quien es el Ingeniero Nivel 2 y cual es su email?", "expected": ["carlos", "ruiz", "@technova.com"], "category": "identificadores"},

    # ======================
    # Incidente crítico
    # ======================
    {"question": "Cuales son los pasos iniciales ante una incidencia critica?", "expected": ["clasificar", "impacto", "urgencia"], "category": "procedimientos"},
    {"question": "Que informacion minima debe registrarse en un ticket critico?", "expected": ["descripcion", "impacto", "urgencia", "evidencias"], "category": "procedimientos"}
]
