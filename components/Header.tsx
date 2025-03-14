import { Button, ThemeProvider, useLayoutContext } from "@gravity-ui/uikit";
import Image from "next/image";

import nebiusLogo from "../public/nebius-logo.svg";
import styles from "./Header.module.css";

export const Header = () => {
  const { isMediaActive } = useLayoutContext();

  return (
    <ThemeProvider theme="dark" scoped rootClassName={styles.root}>
      <header className={styles.header}>
        <span className={styles.left}>
          {isMediaActive("l") && "Powered with"}
          <Image src={nebiusLogo} width={102} height={37} className={styles.logo} unoptimized priority alt="Nebius" />
        </span>
        <nav className={styles.right}>
          <ul className={styles.navigationList}>
            <li>
              <Button
                view="outlined-contrast"
                size={isMediaActive("l") ? "xl" : "m"}
                pin="circle-circle"
                href="https://nebius.com/services/studio-inference-service"
                target="_blank"
              >
                Build your AI app
              </Button>
            </li>
          </ul>
        </nav>
      </header>
    </ThemeProvider>
  );
}; 