/// <reference types="vite/client" />

declare module 'ant-design-vue' {
  const Antd: import('vue').Plugin
  export default Antd
  export const message: {
    success: (content: unknown) => void
    error: (content: unknown) => void
    warning: (content: unknown) => void
    info: (content: unknown) => void
    loading: (content: unknown) => void
  }
}

declare module 'ant-design-vue/dist/reset.css'
