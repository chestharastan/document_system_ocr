/** @type {import('next').NextConfig} */
const nextConfig = {
  // The OCR controller integrates imperative DOM animation in a single mount
  // effect; double-invoking effects in dev StrictMode would attach duplicate
  // listeners, so it is disabled.
  reactStrictMode: false,
};

export default nextConfig;
