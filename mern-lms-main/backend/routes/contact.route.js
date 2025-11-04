import { Router } from "express";
import { verifyToken } from "../utils/verifyUser.js";
import {
  getAllContact,
  getContactsForDMList,
  searchContacts,
  searchChattedContacts,
} from "../controllers/contact.controller.js";

const contactRoute = Router();

contactRoute.post("/search", verifyToken, searchContacts);
contactRoute.get("/get-contacts", verifyToken, getContactsForDMList);
contactRoute.get("/get-all-contacts", verifyToken, getAllContact);
contactRoute.post("/get-chatted-contacts", verifyToken, searchChattedContacts);

export default contactRoute;
