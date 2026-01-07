# Sistema Interno de Soporte – TechNova Solutions

Este repositorio contiene la documentación y configuraciones necesarias para operar el sistema interno de soporte técnico. El objetivo principal es garantizar una gestión eficiente de incidencias, cambios y solicitudes, siguiendo las mejores prácticas del sector.

## Arquitectura del Sistema
El sistema está compuesto por una plataforma central de tickets, módulos de monitorización, herramientas de diagnóstico automático y paneles de KPIs. Cada componente interactúa a través de APIs internas.

## Flujo de Escalado
1. Recepción del ticket por Nivel 1.
2. Clasificación por severidad e impacto.
3. Ejecución de troubleshooting básico.
4. Escalado automático o manual según SLA.
5. Resolución final y documentación del caso.

## Roles del Equipo
### Agente Nivel 1
- Primer punto de contacto.
- Recopila información inicial.
- Ejecuta pruebas simples.

### Agente Nivel 2
- Diagnóstico técnico avanzado.
- Profundización en fallos de hardware/software.

### Especialista Nivel 3
- Resolución de incidentes críticos.
- Trabajo directo con proveedores o desarrollo interno.

## Pasos de Troubleshooting
1. Verificar disponibilidad del servicio.
2. Confirmar credenciales del usuario.
3. Ejecutar comandos básicos de diagnóstico.
4. Revisar logs asociados.
5. Replicar el error en entorno controlado.
6. Documentar cada paso realizado.

## Ejemplo de comando de diagnóstico
```bash
tn_diag --service core_api --mode deep --export /tmp/reporte_diag.json
```