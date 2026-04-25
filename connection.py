import struct
import logging
from time import sleep, time
from typing import Dict
from pymodbus.client import ModbusSerialClient
from pymodbus.exceptions import ModbusException
logging.basicConfig()
log = logging.getLogger()


class ModbusClient:

    ERROR_MAP_DELTA: Dict[int, str] = {
        0x0000: "No message",
        0x1000: "System started",
        0x1001: "StartDefaultTask started",
        0x1002: "StartLimitSwitch started",
        0x1003: "StartListenCommand started",
        0x1004: "KEY1 old-action (legacy)",
        0x1005: "KEY2 pressed",
        0x1006: "KEY3 pressed",
        0x1007: "KEY4 pressed",
        0x1008: "KEY5 pressed",
        0x1009: "Trajectory started",
        0x1010: "Go home / channel command accepted",
        0x1011: "Hard stop command accepted",
        0x1012: "Calibration started",
        0x1013: "Calibration finished",
        0x1014: "Axis counters loaded from Flash",
        0x1015: "Axis counters saved to Flash",
        0x1016: "Axis counters save after calibration OK",
        0x1017: "Move to work pose (0,0,-180) started",
        0x1018: "Already at work pose (0,0,-180)",
        0x1100: "Limit switch event: ch0",
        0x1101: "Limit switch event: ch1",
        0x1102: "Limit switch event: ch2",
        0x1103: "Limit switch event: ch3",
        0x1110: "Buzzer request executed",
        0x1111: "StartBuzzer started",
        0x1222: "All three limit switches active",
        0x2000: "Segment repeat too small",
        0x2001: "Angle exceeds limit",
        0x2002: "Profile overwrite warning",
        0x2012: "Calibration request ignored (busy)",
        0x2014: "Axis counters not valid in Flash (reset to zero)",
        0x2016: "Axis counters save after calibration failed",
        0x2017: "Move to work pose (0,0,-180) rejected",
        0x2011: "Hard stop command rejected",
        0x3017: "Move top center delta overflow",
        0x3001: "Exceeded max step frequency",
        0x4000: "Read: receive CRC bytes failed",
        0x4001: "Read: CRC mismatch",
        0x4002: "Read: register out of range",
        0x4003: "Write: receive bytecount failed",
        0x4004: "Write: packet too large",
        0x4005: "Write: receive data+CRC failed",
        0x4006: "Write: CRC mismatch",
        0x4007: "Write: byte count mismatch",
        0x4008: "Unsupported Modbus function",
        0x4012: "Axis counters Flash write/erase failed",
        0x4013: "Axis counters Flash verify failed",
        0x4110: "Buzzer queue full",
        0x4111: "Buzzer queue not initialized",
    }

    def __init__(self, port, baudrate=115200, slave_addr=0x01, waiting_timeout=10):
        client = ModbusSerialClient(
            port=port,
            baudrate=baudrate,
            parity='N',
            stopbits=1,
            bytesize=8,
            timeout=1,
        ) 
        self.slave_addr = slave_addr
        self.port = port
        
        if not client.connect():
            raise ConnectionError(f"Не удалось открыть порт {port}")
        
        self.client = client
        self.max_regs_per_batch = 120  # <= 123 по Modbus RTU стандарту
        self.max_segments = 200
        self.waiting_timeout = waiting_timeout
    
    def disconnect(self):
        self.client.close()
    
    def rotate(self, angle1, angle2, angle3, angle4=0, time=3000, cmd=1):
        values = [angle1, angle2, angle3, angle3]
        for i in range(4):
            values[i] = round(values[i] * 60)
        values.extend([time, cmd])

        regs = []
        for i in range(4):
            regs.extend(self.int64_to_regs(values[i]))

        # Если после шагов ещё есть доп. команды (например время или команда), дописываем их:
        if len(values) > 4:
            for val in values[4:]:
                val16 = val & 0xFFFF
                regs.append(val16)

        result = self.client.write_registers(
            address=0,
            values=regs,
            slave=self.slave_addr
        )
        if result.isError():
            print("[ERROR] Ошибка записи:", result)
        else:
            print("[OK] Регистры успешно записаны!")

    def driver_config(self, reductor, mode):
        values = [reductor, 0, mode]
        result = self.client.write_registers(
            address=0,
            values=values,
            slave=self.slave_addr
        )
        sleep(0.1)
        if result.isError():
            print("[ERROR] Конфигрурация выполнено:", result)
        else:
            print("[OK] Регистры успешно записаны!")

    def int64_to_regs(self, value):
        """Преобразует int64_t число в 4 Modbus регистра по 16 бит."""
        packed = struct.pack('>q', value)  # '>q' — big-endian signed 64-bit
        regs = []
        for i in range(0, 8, 2):
            reg = (packed[i] << 8) | packed[i + 1]
            regs.append(reg)
        return regs
    
    def is_ready(self):
        ready_registers = self.read(start_addr=6, count=1)  # motion status register
        if isinstance(ready_registers, list) and ready_registers:
            return int(ready_registers[0]) == 1
        return False
    
    def calibrate(self, to_cube=False):
        result = self.client.write_registers(address=406, values=[1], slave=self.slave_addr)
        if result.isError():
            print("[ERROR] failed command calibrate not sent:", result)
            return False
        print("[OK] command calibrate sent.")
        if to_cube:
            start_time = time()
            while not self.is_ready():
                print("[INFO] Waiting for calibration to complete...")
                sleep(0.1)
                if time() - start_time > self.waiting_timeout:
                    print("[ERROR] Timeout waiting for calibration to complete.")
                    return False
            print("[INFO] Moving to cube after calibration...")
            self.move_to_work_top_center()
        return True
    
    def move_to_work_top_center(self):
        result = self.client.write_registers(address=407, values=[1], slave=self.slave_addr)
        if result.isError():
            print("[ERROR] Failed to start move to work pose (0,0,-180)")
            return False
        print("[OK] Move to work pose (0,0,-180) command sent")
        return True
    
    def write_segments(self, segments):
        """
        Отправляет сегменты в контроллер.
        segments: список сегментов, каждый сегмент - список углов для двигателей.
        """
        if len(segments) > self.max_segments:
            log.error(f"Too many segments for firmware: {len(segments)} > {self.max_segments}. Downsample trajectory before send.")
            return False
        registers = []
        for segment in segments:
            for motor_angle in segment:
                # Упаковка int16 → 2 байта → big-endian
                packed = struct.pack('>h', motor_angle)
                registers.append((packed[0] << 8) | packed[1])  # int from bytes

        for offset in range(0, len(registers), self.max_regs_per_batch):
            batch = registers[offset:offset + self.max_regs_per_batch]
            result = self.client.write_registers(address=400 + offset, values=batch, slave=self.slave_addr)
            if result.isError():
                log.error(f"Ошибка при отправке сегментов (offset={offset})")
                return False
            log.info(f"✅ Отправлено: {len(batch)} регистров (offset={offset})")

        log.info("🎯 Все сегменты успешно отправлены.")
        return True

    def proccess(self, cmd=1, duration=1000):
        duration = duration/2
        result = self.client.write_registers(
            address=302,
            values=[int(cmd), int(duration)],
            slave=self.slave_addr,
        )
        if result.isError():
            print("Ошибка запуска")
        else:
            print("Процесс запущен!")
        return result

    def start_trajectory(self, segments, duration=5000):
        """
        Отправляет сегменты в контроллер и запускает их.
        segments: список сегментов, каждый сегмент - список углов для двигателей.
        """
        if self.write_segments(segments):
            launch_payload = struct.pack('>HH', len(segments), duration)
            launch_registers = list(struct.unpack('>2H', launch_payload))

            # Отправка в регистр 300
            result = self.client.write_registers(address=300, values=launch_registers, slave=self.slave_addr)
            if result.isError():
                print("Ошибка запуска")
            else:
                print("Траектория запущена!")

        else:
            log.error("Ошибка при запуске.")
            return False
    
    def _move_to_start(self):
        # Поднимаем вверх
        trajectory = [
            [5100, 5100, 5100]
        ]
        self.start_trajectory(trajectory, duration=2000)

    def registers_description(self, regs):
        pass

    def read(self, start_addr=0, count=10):
        result = self.client.read_holding_registers(address=start_addr, count=count, slave=self.slave_addr)
        if result.isError():
            print("Ошибка чтения:", result)
            return None
        return result.registers
