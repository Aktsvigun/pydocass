/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  webpack(config) {
    // This is needed for @gravity-ui/icons
    config.module.rules.push({
      test: /\.svg$/,
      use: ['@svgr/webpack'],
    });

    // Fix for @gravity-ui/uikit CSS imports
    const oneOfRule = config.module.rules.find(
      (rule) => typeof rule.oneOf === 'object'
    );
    
    if (oneOfRule) {
      const cssRule = oneOfRule.oneOf.find(
        (rule) => rule.test && rule.test.toString().includes('css')
      );
      
      if (cssRule) {
        // Modify issuer to exclude gravity-ui components
        if (cssRule.issuer && cssRule.issuer.and) {
          const gravityUIExclusion = {
            not: [{ test: /node_modules\/@gravity-ui\/uikit/ }],
          };
          cssRule.issuer.and.push(gravityUIExclusion);
        }
      }
    }
    
    return config;
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:4000/:path*',
      },
    ];
  },
  // Transpile the gravity-ui modules
  transpilePackages: ['@gravity-ui/uikit', '@gravity-ui/icons'],
};

module.exports = nextConfig; 