import { render } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

import { App } from "../app/App";

export function renderApp(path = "/overview") {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}
