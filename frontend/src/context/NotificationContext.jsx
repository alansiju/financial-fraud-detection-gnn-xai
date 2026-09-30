import { createContext, useContext, useState, useCallback, useMemo } from 'react';

const NotificationContext = createContext(null);

let _id = 0;

export function NotificationProvider({ children }) {
  const [toasts, setToasts] = useState([]);

  const remove = useCallback((id) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  const add = useCallback((message, type = 'info', duration = 4000) => {
    const id = ++_id;
    setToasts(prev => [...prev, { id, message, type }]);
    if (duration > 0) {
      setTimeout(() => remove(id), duration);
    }
    return id;
  }, [remove]);

  // Memoised so `notify` is the same object on every render. It used to be a
  // fresh object each time, which is why every page left it out of its effect
  // dependencies: adding it would have looped (toast -> provider re-renders ->
  // new notify -> effect re-runs -> refetch -> toast ...). Now it is stable, so
  // listing it as a dependency is both correct and harmless.
  const notify = useMemo(() => ({
    success: (msg, d) => add(msg, 'success', d),
    error:   (msg, d) => add(msg, 'error', d),
    warning: (msg, d) => add(msg, 'warning', d),
    info:    (msg, d) => add(msg, 'info', d),
  }), [add]);

  const value = useMemo(() => ({ toasts, notify, remove }), [toasts, notify, remove]);

  return (
    <NotificationContext.Provider value={value}>
      {children}
    </NotificationContext.Provider>
  );
}

export const useNotification = () => useContext(NotificationContext);
