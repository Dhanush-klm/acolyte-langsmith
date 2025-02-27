import type { NextConfig } from "next";

const nextConfig = {
  experimental: {
    instrumentationHook: true
  }
};

export default nextConfig;
