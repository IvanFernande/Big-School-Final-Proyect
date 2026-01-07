# Procedimiento de Guardia (On-Call)

## Objetivo
Garantizar la disponibilidad operativa del sistema fuera del horario laboral, asignando responsabilidades claras al técnico de guardia.

## Alcance
Aplica a todos los servicios considerados críticos por TechNova Solutions, incluyendo API Core, autenticación, balanceadores y bases de datos principales.

## Horario de On-Call
- Lunes a viernes: 18:00 – 08:00  
- Fines de semana y festivos: 24h

## Responsabilidades del Técnico On-Call
1. Responder alertas en menos de 10 minutos.
2. Registrar toda acción realizada en el sistema de incidencias.
3. Escalar al equipo especializado cuando la causa excede sus atribuciones.
4. Activar protocolo de emergencia si el servicio cae completamente.

## Flujos de Actuación

### Alerta de Servicio Degradado
1. Verificar logs de latencia.
2. Reiniciar el microservicio afectado si procede.
3. Evaluar dependencia con otros módulos.
4. Actualizar estado en el panel interno.

### Alerta de Caída Total
1. Validar con herramientas de monitoreo (tn-monitor).
2. Aislar el nodo defectuoso.
3. Escalar inmediatamente al Nivel 3.
4. Activar redundancia si está disponible.

## Comandos de referencia
```bash
tn-monitor --service core_api --check-health
tn-restart --node backend-07
tn-alert --push "Escalado Nivel 2 requerido"
```