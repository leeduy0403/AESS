import NewDM from "./components/new-dm";
import CreateChannel from "./components/create-channel";
import { useSelector, useDispatch } from "react-redux";
import {
  setDirectMessagesContacts,
  setChannels,
  addMessage,
} from "@/redux/chat/chatSlice";
import { Input } from "@/components/ui/input";
import { useState, useEffect } from "react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { apiClient } from "@/lib/api-client";
import {
  GET_DM_CONTACTS_ROUTE,
  GET_CHANNEL_ROUTE,
  GET_CHATTED_CONTACTS_ROUTE,
} from "@/utils/constants";
import ContactList from "@/components/contact-list";

function ContactsContainer() {
  const { directMessagesContacts, channels, selectedChatType } = useSelector(
    (state) => state.chat
  ); // list of already created contacts
  const dispatch = useDispatch();

  const [chatType, setChatType] = useState("private");
  const [searchedContacts, setSearchedContacts] = useState([]);

  useEffect(() => {
    console.log("!!! ContactsContainer mounted !!!"); //? debug
    const getContacts = async () => {
      const response = await apiClient.get(GET_DM_CONTACTS_ROUTE, {
        withCredentials: true,
      });
      if (response.data.contacts) {
        // console.log(response.data.contacts); //? debug
        dispatch(setDirectMessagesContacts(response.data.contacts));
        if (response.data.contacts.length <= 0) {
          console.log("No contacts found");
        }
      }
    };
    const getChannels = async () => {
      const response = await apiClient.get(GET_CHANNEL_ROUTE, {
        withCredentials: true,
      });
      if (response.data.channels) {
        dispatch(setChannels(response.data.channels));
        if (response.data.channels.length <= 0) {
          console.log("No channels found");
        }
      }
    };

    getContacts();
    getChannels();
  }, [addMessage, dispatch]); //! need testing throughoutly

  //TODO: fix backend
  const searchChattedContacts = async (searchTerm) => {
    try {
      if (searchTerm.length > 0) {
        const response = await apiClient.post(
          GET_CHATTED_CONTACTS_ROUTE,
          { searchTerm },
          { withCredentials: true }
        );
        if (response.data.contacts) {
          console.log(response.data.contacts); //? debug
          dispatch(setDirectMessagesContacts(response.data.contacts));
          if (response.data.contacts.length <= 0) {
            console.log("No contacts found");
          }
        }
      } else {
        setSearchedContacts([]);
      }
    } catch (error) {
      console.log(error);
    }
  };

  return (
    <div className="relative w-full border-2 border-[#2f303b] bg-white md:w-[35vw] lg:w-[35vw] xl:w-[22vw] rounded-md flex-shrink-0">
      <div className="p-3 bg-[#26597C]">
        <div className="flex justify-center items-center px-5">
          <span className="text-3xl font-medium text-white">Chat</span>
        </div>
        <div className="flex justify-between items-center pt-5">
          <Input
            placeholder="Search"
            onChange={(e) => searchChattedContacts(e.target.value)}
          />
          <NewDM />
          <CreateChannel />
        </div>
      </div>
      <Tabs
        defaultValue={
          selectedChatType === "contact"
            ? "private"
            : selectedChatType === "channel"
            ? "group"
            : "private"
        }
        className="w-full"
      >
        <TabsList className="w-full rounded-none bg-transparent p-0">
          <TabsTrigger
            value="private"
            className="w-full h-full rounded-none p-3 overflow-hidden text-black text-lg transition-all duration-300 data-[state=active]:bg-[#F8F8D5] data-[state=active]:font-semibold data-[state=active]:text-black border-b-black border-r-black border-r border-b"
            onClick={() => setChatType("private")}
          >
            Private
          </TabsTrigger>
          <TabsTrigger
            value="group"
            className="w-full h-full rounded-none p-3 overflow-hidden text-black text-lg transition-all duration-300 data-[state=active]:bg-[#F8F8D5] data-[state=active]:font-semibold data-[state=active]:text-black border-b-black border-b"
            onClick={() => setChatType("group")}
          >
            Group
          </TabsTrigger>
        </TabsList>
        <TabsContent value="private" className="mt-0">
          <div className="scrollbar-hidden no-scrollbar max-h-[100vh] overflow-y-auto">
            <ContactList contacts={directMessagesContacts} />
          </div>
        </TabsContent>
        <TabsContent value="group" className="mt-0">
          <div className="scrollbar-hidden no-scrollbar max-h-[100vh] overflow-y-auto">
            <ContactList contacts={channels} isChannel={true} />
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default ContactsContainer;

const Title = ({ text }) => {
  return (
    <h6 className="pl-10 text-base font-normal uppercase tracking-wider">
      {text}
    </h6>
  );
};
