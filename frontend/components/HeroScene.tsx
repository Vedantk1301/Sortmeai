"use client";

import { Canvas } from "@react-three/fiber";
import { Environment, Float, Lightformer, OrbitControls } from "@react-three/drei";
import { useMemo } from "react";

function Ribbon() {
  const geometryArgs = useMemo(() => [1.3, 0.38, 160, 24], []);
  return (
    <Float speed={1.3} rotationIntensity={0.6} floatIntensity={1.4}>
      <mesh position={[0.4, -0.3, 0]} rotation={[Math.PI / 4.5, -Math.PI / 8, 0]}>
        {/* @ts-ignore */}
        <torusKnotGeometry args={geometryArgs} />
        <meshStandardMaterial
          color="#5d7df5"
          metalness={0.55}
          roughness={0.2}
          envMapIntensity={1.2}
        />
      </mesh>
    </Float>
  );
}

function Crystal() {
  return (
    <Float speed={1.1} rotationIntensity={0.4} floatIntensity={1.1}>
      <mesh position={[-1.2, 0.8, 0.5]} castShadow>
        <icosahedronGeometry args={[0.9, 1]} />
        <meshStandardMaterial
          color="#ca337c"
          metalness={0.48}
          roughness={0.18}
          emissive="#ff7aa0"
          emissiveIntensity={0.12}
        />
      </mesh>
    </Float>
  );
}

function SilkyPlane() {
  return (
    <Float speed={0.9} rotationIntensity={0.24} floatIntensity={0.5}>
      <mesh position={[0.6, 0.2, -0.6]} rotation={[Math.PI / 2.4, 0.2, 0]}>
        {/* @ts-ignore */}
        <planeGeometry args={[5, 3, 32, 32]} />
        <meshStandardMaterial
          color="#f8ede3"
          metalness={0.1}
          roughness={0.5}
          wireframe={false}
        />
      </mesh>
    </Float>
  );
}

export function HeroScene() {
  return (
    <Canvas camera={{ position: [0, 0, 6], fov: 38 }}>
      <color attach="background" args={["#f8f4ed"]} />
      <ambientLight intensity={0.65} />
      <directionalLight position={[4, 6, 6]} intensity={1.2} castShadow />
      <spotLight position={[-5, 4, 4]} angle={0.3} intensity={1.1} />

      <Ribbon />
      <Crystal />
      <SilkyPlane />

      <Environment resolution={64}>
        <Lightformer
          form="ring"
          intensity={2}
          color="#f3c9db"
          position={[-10, 2, 5]}
          scale={12}
        />
        <Lightformer form="rect" intensity={1.4} color="#96a9ff" position={[4, -2, 8]} />
      </Environment>
      <OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={0.7} />
    </Canvas>
  );
}
