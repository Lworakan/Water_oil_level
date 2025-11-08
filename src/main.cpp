#include <Arduino.h>

// XKC-Y25 PNP non-contact water level sensor
// Digital output: HIGH when liquid detected, LOW when no liquid
#define SENSOR_PIN 2

void setup() {
  Serial.begin(9600);
  pinMode(SENSOR_PIN, INPUT_PULLDOWN); // Use pull-down resistor

  Serial.println("XKC-Y25 Water/Oil Level Sensor - Basic Reading");
  Serial.println("Sensor connected to GPIO2");
  Serial.println("-------------------------------------------");
  Serial.println("If readings are still inverted, check wiring");
}

void loop() {
  int sensorState = digitalRead(SENSOR_PIN);

  // Invert the logic: PNP sensors often output LOW when detecting
  if (sensorState == LOW) {
    Serial.println("No liquid detected");
    // Serial.println("Liquid DETECTED");
  } else {
    Serial.println("Liquid DETECTED");
  }

  delay(500); // Check every 500ms
}
