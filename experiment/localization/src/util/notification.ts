import { ElNotification } from 'element-plus';
import { ElMessage } from 'element-plus';

export enum NotificationTypes {
  Success = 'success',
  Warning = 'warning',
  Info = 'info',
  Error = 'error'
}

export enum NotificationPositions {
  TopRight = 'top-right',
  TopLeft = 'top-left',
  BottomRight = 'bottom-right',
  BottomLeft = 'bottom-left'
}

export enum MessageTypes {
  Success = 'success',
  Warning = 'warning',
  Error = 'error'
}

export const notify = (
  title: string,
  message: string,
  type: NotificationTypes | undefined,
  position: NotificationPositions | undefined
) => ElNotification({ title, message, type, position });

export const nWarn = (
  title: string,
  message: string,
  position: NotificationPositions | undefined = NotificationPositions.TopRight
) => {
  notify(title, message, NotificationTypes.Warning, position);
};

export const nInfo = (
  title: string,
  message: string,
  position: NotificationPositions | undefined = NotificationPositions.TopRight
) => {
  notify(title, message, NotificationTypes.Info, position);
};

export const nError = (
  title: string,
  message: string,
  position: NotificationPositions | undefined = NotificationPositions.TopRight
) => {
  notify(title, message, NotificationTypes.Error, position);
};

export const nSuccess = (
  title: string,
  message: string,
  position: NotificationPositions | undefined = NotificationPositions.TopRight
) => {
  notify(title, message, NotificationTypes.Success, position);
};

export const mMessage = (message: string, showClose = true) => {
  ElMessage({
    showClose,
    message,
  });
};

export const mSuccess = (message: string, showClose = true) => {
  ElMessage({
    showClose,
    message,
    type: MessageTypes.Success,
  });
};

export const mWarn = (message: string, showClose = true) => {
  ElMessage({
    showClose,
    message,
    type: MessageTypes.Warning,
  });
};

export const mError = (message: string, showClose = true) => {
  ElMessage({
    showClose,
    message,
    type: MessageTypes.Error,
  });
};
