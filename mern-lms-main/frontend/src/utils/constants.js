export const HOST = import.meta.env.VITE_SERVER_URL;

export const CONTACTS_ROUTES = "/api/contact";
export const SEARCH_CONTACTS_ROUTE = CONTACTS_ROUTES + "/search";
export const GET_DM_CONTACTS_ROUTE = CONTACTS_ROUTES + "/get-contacts";
export const GET_ALL_CONTACTS_ROUTE = CONTACTS_ROUTES + "/get-all-contacts";
export const GET_CHATTED_CONTACTS_ROUTE =
  CONTACTS_ROUTES + "/get-chatted-contacts";

export const MESSAGES_ROUTES = "/api/message";
export const GET_ALL_MESSAGES_ROUTE = MESSAGES_ROUTES + "/get-messages";
export const UPLOAD_FILE_ROUTE = MESSAGES_ROUTES + "/upload-file";
export const PIN_MESSAGE_ROUTE = MESSAGES_ROUTES + "/pin-message";
export const UNPIN_MESSAGE_ROUTE = MESSAGES_ROUTES + "/unpin-message";
export const GET_PINNED_MESSAGES_ROUTE =
  MESSAGES_ROUTES + "/get-pinned-messages";
export const GET_LATEST_MESSAGES_ROUTE =
  MESSAGES_ROUTES + "/get-latest-message";

export const CHANNEL_ROUTES = "/api/channel";
export const CREATE_CHANNEL_ROUTE = CHANNEL_ROUTES + "/create-channel";
export const GET_CHANNEL_ROUTE = CHANNEL_ROUTES + "/get-channel";
export const SEARCH_CHANNEL_ROUTE = CHANNEL_ROUTES + "/search-channel";
export const GET_CHANNEL_MESSAGES = CHANNEL_ROUTES + "/get-channel-messages";
export const CHANGE_CHANNEL_NAME = CHANNEL_ROUTES + "/change-channel-name";
export const GET_CHANNEL_LATEST_MESSAGES_ROUTE =
  CHANNEL_ROUTES + "/get-channel-latest-message";
export const GET_USER_IN_CHANNEL_ROUTE =
  CHANNEL_ROUTES + "/get-user-in-channel";
export const SEARCH_USER_NOT_IN_CHANNEL_ROUTE =
  CHANNEL_ROUTES + "/search-user-not-in-channel";
export const GET_CHANNEL_PINNED_MESSAGES_ROUTE =
  CHANNEL_ROUTES + "/get-channel-pinned-messages";
